__author__ = 'Bhushan Kotnis'

import unittest
from kge.models import get_model
import numpy as np

class TestModels(unittest.TestCase):

    def setUp(self):
        print("Setup")

    def tearDown(self):
        print("Tear down")

    def sigmoid(self,x):
        return np.exp(x)/( 1.0 + np.exp(x))

    def max_margin(self,scores):
        margin = np.maximum(0,1-scores[0] + scores[1:])
        return np.sum(margin)

    def numpy_bilinear_model(self,x_s,x_t,W):
        scores = []
        for u,v in zip(x_s,x_t):
            scores.append(np.dot(np.dot(W,v),np.transpose(u)))
        scores = np.asarray(scores)
        score = scores[0]
        cost = self.max_margin(scores)
        return score,cost

    def numpy_s_bilinear_model(self,X_s,X_t,x_r,W):
        scores = []
        for u,v in zip(X_s,X_t):
            outer = np.reshape(np.outer(u,v),(1,-1))[0,:]
            z = np.dot(W,outer)
            scores.append(np.dot(z,x_r))
        cost = self.max_margin(scores)
        return scores[0],cost

    def test_s_bilinear_model(self):
        dim = 10
        r_dim = 10
        num_negs = 8
        X_s = []
        X_t = []
        for i in range(num_negs):
            X_s.append(0.1*np.random.uniform(size = [dim]))
            X_t.append(0.1*np.random.uniform(size = [dim]))

        W = np.random.uniform(size = [r_dim,dim*dim])
        x_r = np.random.uniform(size=[r_dim])

        np_score,np_cost = self.numpy_s_bilinear_model(X_s,X_t,x_r,W)
        gi_model = get_model("group interaction")
        score = gi_model['score']
        fprop = gi_model['fprop']
        X_s = np.asarray(X_s)
        X_t = np.asarray(X_t)

        self.assertAlmostEqual(np.linalg.norm(score(X_s,X_t,W,x_r)-np_score),0.0,delta=0.0001)
        self.assertAlmostEqual(fprop(X_s,X_t,W,x_r), np_cost, delta=0.0001)


    def test_bilinear_model(self):

        dim = 10
        num_negs = 8
        X_s = 0.1*np.random.normal(size=[dim])
        X_t = []
        for i in range(num_negs):
            X_t.append(0.1*np.random.normal(size=[dim]))
        W = np.random.normal(size=[dim,dim])

        np_score, np_fprop_1 = self.numpy_bilinear_model(X_s,X_t,W)

        bilinear = get_model("bilinear")
        score = bilinear['score']
        fprop = bilinear['fprop']
        X_s = np.asarray(X_s)
        X_t = np.asarray(X_t)

        self.assertAlmostEqual(np.linalg.norm(score(X_s,X_t,W)-np_score),0.0,delta=0.0001)

        self.assertAlmostEqual(fprop(X_s,X_t,W), np_fprop_1, delta=0.0001)


    def test_coupling(self):
        dim = 10
        X_s = []
        X_t = []
        W_p = []
        np_results = []
        for i in range(5):
            X_s.append(np.random.normal(size= [dim]))
            X_t.append(np.random.normal(size= [dim]))
            W_p.append(np.random.normal(size= [dim,dim]))
            np_results.append(self.sigmoid(np.dot(np.dot(W_p[i],X_t[i]),np.transpose(X_s[i]))))

        from kge.theano_models import test_coupling
        fprop = test_coupling()
        result = fprop(np.asarray(X_s),np.asarray(X_t),np.asarray(W_p))

        self.assertAlmostEqual(np.linalg.norm(np.asarray(np_results)-result),0.0,delta=0.0001)

    def test_nn(self):
        dim = 5
        x = np.random.uniform(0,1,dim)
        W_h = np.random.normal(size=[dim,dim])
        b = np.random.normal()
        from kge.theano_models import test_nn
        fprop = test_nn()
        h = fprop(x,W_h,b)
        h_np = self.sigmoid(np.dot(W_h,x) + b)
        self.assertAlmostEqual(np.linalg.norm(h-h_np),0.0,delta=0.0001)


    def cross_entropy(self,p,q):
        return -np.dot(p,np.log(q))

    def test_ordistic(self):
        dim = 5
        num_classes = 5
        h = np.random.uniform(0,1,dim)
        u = np.random.uniform(size=(1,num_classes))
        w_o = np.random.normal(size=dim)
        c = np.random.normal()
        y = np.zeros([1,num_classes])
        y[0,0] = 1

        # numpy
        z = np.dot(w_o.transpose(),h) + c
        q = np.exp(u*z + (y[0,:] - np.square(u)/2))
        np_score = q/q.sum()
        self.assertAlmostEqual(np_score.sum(),1.0,msg="Numpy scores not normalized",delta=0.0001)
        np_cost = self.cross_entropy(y[0,:],np_score[0,:])

        from kge.theano_models import test_ordistic
        fprop = test_ordistic()
        score,cost = fprop(h,w_o,u,c,y)
        score = score[0,:]
        self.assertAlmostEqual(score.sum(),1.0,msg="Theano scores not normalized",delta=0.0001)
        self.assertAlmostEqual(np.linalg.norm(score-np_score),0.00,delta=0.0001)
        self.assertAlmostEqual(np_cost-cost,0.00,delta=0.0001)

