__author__ = 'Bhushan Kotnis'

from parameters import SparseParams
import numpy as np
import constants
import util
from data import Path


class Updater(object):

    def update(self,grad,t=1.0):
        '''
        Returns update parameters
        :param grad:
        :return: update parameters
        '''
        raise NotImplementedError()

class Adam(Updater):

    def __init__(self,lr=0.001,beta_1=0.9,beta_2=0.999,eps=1e-08):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.grad = SparseParams(d=dict())
        self.grad_sq = SparseParams(d=dict())

    def update(self,grad,t=1.0):
        delta = SparseParams(d=dict())
        for param in grad:
            g_1 = grad[param] * (1 - self.beta_1)
            g_2 = np.power(grad[param],2)*(1-self.beta_2)
            if param in self.grad:
                self.grad[param] = self.grad[param]*self.beta_1 + g_1
                self.grad_sq[param] = self.grad_sq[param]*self.beta_2 + g_2
            else:
                # Add to zeros
                self.grad[param] =  g_1
                self.grad_sq[param] =  g_2

            moment_1 = self.grad[param]* 1.0 / (1 - np.power(self.beta_1,t+1))
            moment_2 =  self.grad_sq[param]* 1.0 / (1 - np.power(self.beta_2,t+1))

            delta[param] = -1.0*self.lr*moment_1 / ( np.sqrt(moment_2) + self.eps )

        return delta


class Adagrad(Updater):
    def __init__(self,lr=0.01,eps=1e-08):
        self.lr = lr
        self.grad_sq = SparseParams(d=dict())
        self.eps = eps

    def update(self,grad,t=1.0):
        delta = SparseParams(d=dict())
        for param in grad:
            g_2 = np.power(grad[param], 2)
            if param in self.grad_sq:
                self.grad_sq[param] = self.grad_sq[param]+ g_2
            else:
                # Add to zeros
                self.grad_sq[param] =  g_2

            delta[param] = -1.0 * self.lr *grad[param]/ (np.sqrt(self.grad_sq[param]) + self.eps)
        return delta


'''
----------------------------------------------------
'''


class Evaluater(object):

    def evaluate(self,params,x):
        raise NotImplementedError()

class RankEvaluater(Evaluater):
    def __init__(self,model,neg_sampler):
        self.neg_sampler = neg_sampler
        self.model = model
        self.num_negs = constants.num_train_negs
        self.init_score = float('inf') # because lower the better
        self.metric_name = "Mean Rank"


    def comparator(self,curr_score,prev_score):
        # write if curr_score less than prev_score
        return curr_score < prev_score

    def evaluate(self,params,batch):
        '''
        Computes mean rank
        :param params: model parameters
        :param batch: data batch
        :return: mean rank 
        '''
        pos_scores = self.model.predict(params,batch).flatten()
        mean_rank = []
        for p,ex in zip(pos_scores,batch):
            negs = self.neg_sampler.sample(ex,self.num_negs,True)
            neg_ex = [Path(ex.s,ex.r,n) for n in negs]
            scores = self.model.predict(params,neg_ex).flatten()
            scores = np.insert(scores,constants.pos_position,p)
            ranks = util.ranks(scores, ascending=False)
            mean_rank.append(ranks[constants.pos_position])

        return np.nanmean(mean_rank)


