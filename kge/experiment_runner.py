
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
    json.dump(config,open(os.path.join(results_dir,'config.json'),'w'),sort_keys=True, separators=(',', ': '))
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
    neg_sampler.count_test_cats(data_set['test'])
    hit_count(data_set['test'],params,model,neg_sampler)
    #evaluate(data_set['test'],params,model,config['model'],neg_sampler,results_dir,
    #         config.get('num_test_negs',constants.num_test_negs),config.get('k',1))


def train_test(config,exp_name,data_path):
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)

#ToDo: Messy code, refactor
def hit_count(data,params,model,neg_sampler):
    hits = 0

    for count,ex in enumerate(data):
        can_cats = list(neg_sampler._test_cats.get(ex.r[0], set()))
        c_max = -float('inf')
        cat = ""
        if len(can_cats) > 0:
            for c in can_cats:
                cat_ex = Path(ex.t,("has_category",),c)
                c_score = model.predict(params, cat_ex)
                if c_score > c_max:
                    c_max = c_score
                    cat = c
            gold_cats = neg_sampler._entity_cats.get(ex.t,set())
            if cat in gold_cats:
                hits += 1
        if count % 1000 == 0:
            print (count)
    print("Category Recall {}".format(float(hits)/len(data)))

def evaluate(data,params,model,model_name,neg_sampler,results_dir,k,num_negs=float('inf')):

    def calc_hits(rank,pos):
        return 1.0 if rank <= pos else 0.0

    def compute_metrics(ex):
        if model_name=='tr':
            can_cats = list(neg_sampler._test_cats.get(ex.r[0],set()))
            c_max = -float('inf')
            cats = ""
            if len(can_cats) > 0:
                for c in can_cats:
                    c_score =  model.attention(params,ex,c)
                    if c_score > c_max:
                        c_max = c_score
                        cats = c
            true_cats = neg_sampler._entity_cats[ex.t]
            print("Chose Category {}, True Categories ".format(cats)+ true_cats)
            cats = [cats]

        else:
            # No Attn model.
            cats = neg_sampler.get_test_cats(ex.r[0])

        candidates = neg_sampler.get_typed_entities(cats)
        if ex.t not in candidates:
            return 0.0, 0.0, np.zeros(3)
        pos = model.predict(params, ex)
        candidates.remove(ex.t)
        if len(candidates) < 1:
            candidates = neg_sampler._entity_cats.keys()
            if ex.t in candidates:
                candidates.remove(ex.t)
            assert len(candidates) > 1

        scores = []
        for ent in candidates:
            p = Path(ex.s,ex.r,ent)
            scores.append(model.predict(params,p))


        scores = np.asarray(scores,dtype=theano.config.floatX).flatten()
        # Average Quantile, score[0] is positive
        avg_quantile =  util.average_quantile(pos,scores)
        rank = util.rank_from_quantile(avg_quantile,scores.shape[0])
        # HITS @ 10
        hits = []
        hits.append(calc_hits(rank,1))
        hits.append(calc_hits(rank, 3))
        hits.append(calc_hits(rank, 10))
        return avg_quantile,1.0/rank,hits

    mean_quantile = 0.0
    hits = np.zeros(3)
    mrr = 0.0
    count = 0
    for d in data:
        avg_quantile, rr, hits_all = compute_metrics(d)
        if not np.isnan(avg_quantile):
            mean_quantile = (mean_quantile*count + avg_quantile)/(count+1)
            for i,h in enumerate(hits_all):
                hits[i] = (hits[i]*count + h)/(count + 1)

            mrr = (mrr*count + rr)/(count+1)
            count += 1
            if count%100==0:
                print("Query Count : {}".format(count))
    print('Writing Results.')
    with open(os.path.join(results_dir,'results'),'w') as f:
        f.write("Mean Quantile : {:.4f}\nMean Reciprocal Rank : {:.4f}\nHITS@1,3,10 : {:.4f}, {:.4f}, {:.4f}\n".
                format(mean_quantile,mrr,hits[0],hits[1],hits[2]))





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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('exp_name')
    args = parser.parse_args()
    main(args.exp_name,args.data_path)


