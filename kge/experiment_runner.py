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
    elif operation=='visualize':
        visualize_relation(config,exp_name,data_path,"/tv/tv_genre/programs")
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
    data_set = data.read_dataset(data_path,dev_mode=is_dev,is_cat=is_cat)

    print("\n***{} MODE***\n".format('DEV'if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    #ToDo:if model is coupled, then first learn single model

    # Set up functions and params
    neg_sampler = data.NegativeSampler(data_set['train'],typed=False,
                                       cats=data.load_categories(constants.cat_file))
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

    neg_sampler = data.NegativeSampler(all_data,typed=False,cats=data.load_categories(constants.cat_file))
    # Initializing the model changes config.

    model = models.get_model(config, neg_sampler)
    params = data.load_params(params_path, model)
    print("Number of Test Samples {}".format(len(data_set['test'])))
    evaluate(data_set['test'],params,model,config['model'],neg_sampler,results_dir,
             config.get('num_test_negs',constants.num_test_negs),config.get('k',1))


def visualize_relation(config,exp_name,data_path,r):
    print("Generating attention vector for relation {}".format(r))
    def get_relation_instances(data):
        relations = dict()
        for ex in data:
            instances = relations.get(ex.r[0],set())
            instances.add((ex.s,ex.t))
            relations[ex.r[0]] = instances
        return relations

    def tuple_attn(cat_attn):
        attn = []
        for c in cat_attn:
            attn.append((c,cat_attn[c]))
        return attn

    def write_attn(cat_attn,f):
        cat_attn.sort(key=lambda tup: tup[1],reverse=True)
        for c in cat_attn:
            f.write("{}\t{}\n".format(c[0],c[1]))
        f.close()

    def compute_attn(relations,params,r):
        cat_attn = dict()
        for e in relations[r]:
            ex = Path(e[0],(r,),e[1])
            attn, cats = model.attention(params, ex)
            arg_max = np.argsort(-1.0 * attn)
            best_cat = cats[arg_max[0]]
            for c in cats:
                if c not in cat_attn:
                    cat_attn[c] = 0.0
            #print(best_cat)
            cat_attn[best_cat] = cat_attn[best_cat] + 1.0
            #print(cat_attn)
        return tuple_attn(cat_attn)


    results_dir = os.path.join(data_path, exp_name)
    params_path = os.path.join(data_path, exp_name, 'params_single.cpkl')
    data_set = data.read_dataset(data_path, dev_mode=True)
    neg_sampler = data.NegativeSampler(data_set['train'], typed=False,
                                       cats=data.load_categories(constants.cat_file))
    model = models.get_model(config, neg_sampler)
    params = data.load_params(params_path, model)
    relations = get_relation_instances(data_set['train'])
    attn = compute_attn(relations,params,r)
    f = open(results_dir+'/attention_{}.tsv'.format(r.replace("/","_")), 'w')
    write_attn(attn, f)



def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)

#ToDo: Messy code, refactor
def evaluate(data,params,model,model_name,neg_sampler,results_dir,num_negs,k=1):

    def compute_metrics(batch):
        batch_size = len(batch)
        avg_quantile = 0.0
        rank = 0
        hits_10 = 0
        if model_name=='tr':
            # Find the top k categories and treat all entities
            # belonging to that category as candidates
            attn,cats = model.attention(params,batch)
            arg_max = np.argsort(-1.0*attn,axis=1)
            arg_max = arg_max[0,:np.minimum(k,len(arg_max))]
            cats = [cats[i] for i in arg_max]
            # if positive not found return 0

        else:
            # No Attn model.
            cats = [neg_sampler.test_cats[ex.r[0]] for ex in batch]

        candidates = [neg_sampler.get_typed_entities(c) for c in cats]
        ind = 0
        batch_targets = []
        for ex,ents in zip(batch,candidates):
            # if actual target not in ents then scores are
            if ex.t in ents:
                targets = [Path(ex.s,ex.r,t) for t in ents]
                batch_targets.append(targets)
                ind += 1
            else:
                del batch[ind]

        pos = model.predict(params,batch)
        for ind,p in enumerate(pos):
            scores = model.predict(params,batch_targets[ind])
            scores = np.insert(scores,0,p)
            scores = np.asarray(scores,dtype=theano.config.floatX).flatten()
            # Average Quantile, score[0] is positive
            avg_quantile =  (avg_quantile*ind + util.average_quantile(np.asarray([scores[0]]),scores[1:]) )/(ind+1)
            rank = (1.0/rank*ind + 1.0/util.rank_from_quantile(avg_quantile,scores.shape[0]) )/(ind+1)
             # HITS @ 10
            hits_10 = (hits_10*ind + 1.0 if 1.0/rank<=10 else 0.0 )/(ind+1)
        #Adjust for missing targets
        avg_quantile = avg_quantile*len(pos)/batch_size
        return avg_quantile*len(pos)/batch_size,(rank*len(pos)/batch_size),hits_10*len(pos)/batch_size

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



