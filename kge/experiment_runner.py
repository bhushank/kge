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
    data_set = data.read_dataset(data_path,dev_mode=is_dev,max_examples=1000)

    print("\n***{} MODE***\n".format('DEV'if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    #ToDo:if model is coupled, then first learn single model

    # Set up functions and params
    neg_sampler = data.NegativeSampler(data_set['train'],typed=True)
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
    if 'params_path' not in config:
        # ToDo:Hardcoded, will be different for coupled
        params_path = os.path.join(data_path,exp_name,'params_single.cpkl')
        if not os.path.exists(params_path):
            print("No trained params found, quitting.")
            return
    else:
        params_path = config['params_path']

    data_set = data.read_dataset(data_path,dev_mode=is_dev)
    all_data = copy.copy(data_set['train'])
    #ToDo:Does not contain dev
    all_data.extend(data_set['test'])
    neg_sampler = data.NegativeSampler(all_data,typed=True)
    model = models.get_model(config, neg_sampler)
    params = data.load_params(params_path, model)
    evaluate(data_set['test'],params,model,neg_sampler,results_dir,
             config.get('num_test_negs',constants.num_test_negs))


def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)


def evaluate(data,params,model,neg_sampler,results_dir,num_negs):

    def compute_metrics(ex):
        scores = []
        scores.append(model.predict(params,ex))
        negs = neg_sampler.sample(ex,num_negs,True)
        for t in negs:
            q = Path(ex.s,ex.r,t)
            scores.append(model.predict(params, q))

        if len(scores)<=1:
            return np.nan,np.nan,np.nan

        scores = np.asarray(scores,dtype=theano.config.floatX).flatten()

        # Average Quantile, score[0] is positive
        avg_quantile = util.average_quantile(np.asarray([scores[0]]),scores[1:])
        rank = util.rank_from_quantile(avg_quantile,scores.shape[0])
        # HITS @ 10
        hits_10 = 1.0 if rank<=10 else 0.0
        return avg_quantile,1.0/rank,hits_10

    mean_quantile = 0.0
    hits_at_10 = 0.0
    mrr = 0.0
    count = 0
    for d in data:
        avg_quantile, rr, hits_10 = compute_metrics(d)
        if not np.isnan(avg_quantile):
            mean_quantile = (mean_quantile*count + avg_quantile)/(count+1)
            hits_at_10 = (hits_at_10*count + hits_10)/(count + 1)
            mrr = (mrr*count + rr)/(count+1)
            count += 1
            if count%100==0:
                print("Query Count : {}".format(count))
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



