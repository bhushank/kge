__author__ = 'Bhushan Kotnis'

import argparse
import json
import os
import data
import algorithms
import optimizer
import util
import time
import numpy as np


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
    data_set = data.read_dataset(data_path,dev_mode=is_dev)
    print("\n***{} MODE***\n".format('DEV'if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    #ToDo:if model is coupled, then first learn single model

    # Set up functions and params
    rel_dimension = config.get('relation_dim',-1)
    entity_dim = config.get('entity_dim')
    neg_sampler = data.NegativeSampler(data_set['train'],data_set['test'], data_path,
                                       config.get('max_neg_train',10))
    objective = algorithms.Objective(config['model'],entity_dim,rel_dimension,neg_sampler,
                                     config['param_scale'],config['l2_reg'])
    evaluater = algorithms.RankEvaluater(objective,neg_sampler,config.get('num_dev_samples',200))

    updater = algorithms.Adam()

    minimizer = optimizer.GradientDescent(data_set['train'],data_set['test'],updater,
                                          objective,evaluater,results_dir,'single',config)
    print('Training...\n')
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
    rel_dimension = config.get('relation_dim', -1)
    entity_dim = config.get('entity_dim')
    neg_sampler = data.NegativeSampler(data_set['train'], data_set['test'], data_path,
                                       200,is_dev=config['is_dev'],is_train = False)
    objective = algorithms.Objective(config['model'], entity_dim, rel_dimension, neg_sampler,
                                     config['param_scale'], config['l2_reg'])
    params = data.load_params(params_path, objective)
    evaluate(data_set['test'],params,objective,results_dir)


def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)


def evaluate(data,params,objective,results_dir):

    def compute_metrics(ex):
        scores = objective.predict(params,ex,200)
        if scores is None:
            return np.nan,np.nan,np.nan
        else:
            if len(scores.shape) > 1:
                scores = np.reshape(scores, (scores.shape[1],))
            # Average Quantile, score[0] is positive
            avg_quantile = util.average_quantile(np.asarray([scores[0]]),scores[1:])
            rank = util.rank_from_quantile(avg_quantile,len(scores))
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
            if count%10000==0:
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



