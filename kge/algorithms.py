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
    def __init__(self,model,neg_sampler,num_samples):
        self.neg_sampler = neg_sampler
        self.model = model
        self.num_samples = num_samples
        self.init_score = 10000 # because lower the better
        self.metric_name = "Mean Rank"

    def comparator(self,curr_score,prev_score):
        # write if curr_score less than prev_score
        return curr_score < prev_score

    def evaluate(self,params,ex):
        pos = self.model.predict(params,ex)
        negs = self.neg_sampler.sample(ex,self.num_samples,True)
        if len(negs) < 1:
            return np.nan
        scores = []
        for n in negs:
            ex = Path(ex.s,ex.r,n)
            scores.append(self.model.predict(params,ex))
        scores.insert(constants.pos_position,pos)
        scores = np.asarray(scores)
        ranks = util.ranks(scores.flatten(),ascending=False)
        if ranks is None:
            return np.nan
        return ranks[constants.pos_position]



