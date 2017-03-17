__author__ = 'Bhushan Kotnis'

import argparse
import json
import os
import data
from data import Path
import algorithms
import models
import optimizer
import util
import time
import numpy as np
import constants
import copy
import theano

def main(exp_name,data_path):
    config = json.load(open(os.path.join(data_path,'experiment_specs',"{}.json".format(exp_name))))

    operation = config.get('operation','train_test')
    if operation=='train':
        train(config,exp_name,data_path)
    elif operation=='test':
        test(config,exp_name,data_path)
    elif operation=='train_test':
        train_test(config,exp_name,data_path)
    else:
        raise NotImplementedError("{} Operation Not Implemented".format(operation))

def train(config,exp_name,data_path):
    # Read train and dev data, set dev mode = True
    results_dir =  os.path.join(data_path,exp_name)
    if os.path.exists(results_dir):
        print("{} already exists, no need to train.\n".format(results_dir))
        return
    os.makedirs(results_dir)
    json.dump(config,open(os.path.join(results_dir,'config.json'),'w'))
    is_dev = True
    is_cat = config['model'] == constants.tr
    data_set = data.read_dataset(data_path,dev_mode=is_dev,max_examples=1000)

    print("\n***{} MODE***\n".format('DEV'if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    # Set up functions and params
    neg_sampler = data.NegativeSampler(data_set['train'])
    model = models.get_model(config,neg_sampler)
    evaluater = algorithms.RankEvaluater(model,neg_sampler,
                                         config.get('num_dev_negs',constants.num_dev_negs))
    updater = algorithms.Adam()
    minimizer = optimizer.GradientDescent(data_set['train'],data_set['test'],updater,
                                          model,evaluater,results_dir,'single',config)
    print('Training {}...\n'.format(config['model']))
    start = time.time()
    minimizer.minimize()
    end = time.time()
    hours = (end-start)/ 3600
    minutes = ((end-start) % 3600) / 60
    print("Finished Training! Took {} hours and {} minutes\n".format(hours,minutes))



def test(config,exp_name,data_path):

    print("Testing...\n")
    is_dev = config['is_dev']
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    results_dir = os.path.join(data_path, exp_name)
    params_path = os.path.join(data_path,exp_name,'params_single.cpkl')
    if not os.path.exists(params_path):
        print("No trained params found, quitting.")
        return

    data_set = data.read_dataset(data_path,dev_mode=is_dev)
    all_data = copy.copy(data_set['train'])

    neg_sampler = data.NegativeSampler(all_data)
    # Initializing the model changes config.

    model = models.get_model(config, neg_sampler)
    params = data.load_params(params_path, model)
    print("Number of Test Samples {}".format(len(data_set['test'])))
    evaluate(data_set['test'],params,model,neg_sampler,results_dir)



def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)


def evaluate(data,params,model, neg_sampler,results_dir):

    def pack_negs(ex,negs):
        batch = []
        for n in negs:
            batch.append(Path(ex.s, ex.r, n))
        return batch

    def compute_metrics(batch):
        avg_quantile = 0.0
        rr = 0
        hits_10 = 0
        ind = 0
        pos = model.predict(params,batch)
        for ex,p in zip(batch,pos):
            negs = pack_negs(ex,neg_sampler.sample(ex,float('inf')))
            scores = model.predict(params,negs)
            scores = np.insert(scores,0,p)
            scores = np.asarray(scores,dtype=theano.config.floatX).flatten()
            # Average Quantile, score[0] is positive
            avg_quantile =  (avg_quantile*ind + util.average_quantile(np.asarray([scores[0]]),scores[1:]) )/(ind+1)
            rank = util.rank_from_quantile(avg_quantile,scores.shape[0])
            rr = (rr*ind + 1.0/rank )/float(ind+1)
             # HITS @ 10
            h_10 = 1.0 if  rank<=10 else 0.0
            hits_10 = (hits_10*ind + h_10)/(ind+1)
            print rank
        return avg_quantile,rr,hits_10


    mean_quantile = 0.0
    hits_at_10 = 0.0
    mrr = 0.0
    count = 0
    for d in util.chunk(data,constants.test_batch_size):
        avg_quantile, rr, hits_10 = compute_metrics(d)
        if not np.isnan(avg_quantile):
            mean_quantile = (mean_quantile*count + avg_quantile)/(count+1)
            hits_at_10 = (hits_at_10*count + hits_10)/float(count + 1)
            mrr = (mrr*count + rr)/(count+1)
            count += 1
            if count%10==0:
                print("Query Count : {}".format(count))
                print("Mean Quantile : {:.4f}, Mean Reciprocal Rank : {:.4f}, HITS@10 : {:.4f}".
                format(mean_quantile,mrr,hits_at_10))

    print('Writing Results.')
    with open(os.path.join(results_dir,'results'),'w') as f:
        f.write("Mean Quantile : {:.4f}\nMean Reciprocal Rank : {:.4f}\nHITS@10 : {:.4f}\n".
                format(mean_quantile,mrr,hits_at_10))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('exp_name')
    args = parser.parse_args()
    main(args.exp_name,args.data_path)



