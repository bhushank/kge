__author__ = 'Bhushan Kotnis'

import numpy as np
import theano
import itertools

def chunk(arr,chunk_size):
    for i in range(0,len(arr),chunk_size):
        yield arr[i:i+chunk_size]


def sample(data,num_samples,replace=False):

    if len(data) <= num_samples:
        return data
    idx = np.random.choice(len(data),num_samples,replace=replace)
    return [data[i] for i in idx]

def pad(x,size):
    if size <=0:
        return x
    padding = np.asarray([x for i in range(size)]).squeeze().T
    # gives (100,) for size 1
    if size==1:
      padding.resize(padding.shape[0],1)
    X_batch = np.append(x, padding, axis=1)
    return X_batch

def memoize(f):
    cache = {}

    def decorated(*args):
        if args not in cache:
            cache[args] = f(*args)
        else:
            print("Loading cached values")

        return cache[args]

    return decorated

def to_floatX(ndarray):
    return np.asarray(ndarray,dtype = theano.config.floatX)


def f1_score(true_positives,positives):
    hits = len([x for x in positives if x in true_positives])

    precision = float(hits)/len(positives)
    recall = float(hits) / len(true_positives)
    return 2.0*precision*recall/(precision + recall)



def ranks(scores, ascending = True):
    sign = 1 if ascending else -1
    idx = np.argsort(sign*scores)
    ranks = np.empty(scores.shape[0],dtype=int)
    ranks[idx] = np.arange(scores.shape[0])
    ranks += 1 # start from 1
    return ranks

def quantile(rank,total):
    '''
    Returns quantile or how far from the top rank / total
    :param rank:
    :param total:
    :return quantile:
    '''
    if total==1:
        return np.nan
    return (total-rank)/float(total-1)

def rank_from_quantile(quantile, total):
    if np.isnan(quantile):
        return 1
    return total - quantile * (total - 1)



# Assumes number of positives and negatives
def average_quantile(positives,negatives):
    all = np.concatenate((positives,negatives))
    # compute ranks of all positives
    all_ranks = ranks(all,False)[:len(positives)]
    # actual positives
    pos_ranks = ranks(positives,False)
    # (p-1) because ranks start from 1. filtered rank needed for more than one positive sample. For one sample, p = 1, r = a
    filtered_ranks = [a - (p-1) for a,p in itertools.izip(all_ranks,pos_ranks)]
    n = len(negatives) + 1 # adjustment for -1 in quantile function
    # n-r will give the number of negatives after the positive for each positive
    quantiles = np.nanmean([quantile(r,n) for r in filtered_ranks])
    return np.nanmean(quantiles)

# For one positive and many negatives
def reciprocal_rank(scores,correct_pos):
    assert correct_pos < scores.shape[0]
    rank =  ranks(scores,ascending=False)
    return 1.0/rank[correct_pos]


def get_correlation_tensor(dim):
    W = np.zeros((dim,dim,dim),dtype=theano.config.floatX)
    for i in range(dim):
        W[i,:,i] = np.ones(dim,dtype=theano.config.floatX)
    return W
