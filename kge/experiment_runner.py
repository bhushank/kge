
__author__ = 'Bhushan Kotnis'

import argparse
import json
import os
import data

import algorithms
import models
import optimizer
import util
import time
import constants
import copy
from evaluation import TestEvaluater,RankEvaluater
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
    json.dump(config,open(os.path.join(results_dir,'config.json'),'w'),
              sort_keys=True,separators=(',\n', ': '))
    is_typed = config.get('is_typed',False)
    print("Typed Regularizer {}".format(is_typed))
    data_set = data.read_dataset(data_path,dev_mode=True,is_typed=is_typed)
    is_dev = config['is_dev']
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    # Set up functions and params
    neg_sampler = data.NegativeSampler(data_set['train'])
    model = models.get_model(config,neg_sampler)
    evaluater = RankEvaluater(model,neg_sampler)
    updater = algorithms.Adam()
    typed_data = data_set['typed'] if is_typed else None
    minimizer = optimizer.GradientDescent(data_set['train'],data_set['test'],updater,
                                          model,evaluater,results_dir,'single',config,is_typed=is_typed,typed_data=typed_data)
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
    all_data.extend(data_set['dev'])
    all_data.extend(data_set['test'])
    neg_sampler = data.NegativeSampler(all_data)
    # Initializing the model changes config.
    model = models.get_model(config, neg_sampler)
    params = data.load_params(params_path, model)
    print("Number of Test Samples {}".format(len(data_set['test'])))
    evaluater = TestEvaluater(model, neg_sampler,params, is_dev, results_dir)
    evaluate(data_set['test'],evaluater,results_dir,is_dev)



def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)


def evaluate(data,evaluater,results_dir,is_dev):
    print("Evaluating")
    h10,mrr = 0.0,0.0
    start = time.time()
    report_period = 1
    for count,d in enumerate(util.chunk(data,constants.test_batch_size)):
        rr, hits_10 = evaluater.evaluate(d,constants.num_test_negs)
        h10 = (h10*count + hits_10)/float(count + 1)
        mrr = (mrr*count + rr)/(count+1)
        if count%report_period==0:
            end = time.time()
            secs = (end - start)
            print("Speed {} queries per second".format(report_period*constants.test_batch_size/float(secs)))
            print("Query Count : {}".format(count))
            print("Mean Reciprocal Rank : {:.4f}, HITS@10 : {:.4f}".
            format(mrr,h10))
            start = time.time()

    print('Writing Results.')
    split = 'dev' if is_dev else 'test'
    all_ranks = [str(x) for x in evaluater.all_ranks]
    with open(os.path.join(results_dir,'ranks_{}'.format(split)),'w') as f:
        f.write("\n".join(all_ranks))
    with open(os.path.join(results_dir,'results_{}.'.format(split)),'w') as f:
        f.write("Mean Reciprocal Rank : {:.4f}\nHITS@10 : {:.4f}\n".
                format(mrr,h10))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('exp_name')
    args = parser.parse_args()
    main(args.exp_name,args.data_path)


