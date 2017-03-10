__author__ = 'Bhushan Kotnis'

import unittest
from kge.theano_models import get_model
import numpy as np

class TestModels(unittest.TestCase):

    def setUp(self):
        print("Setup")

    def tearDown(self):
        print("Tear down")

    def sigmoid(self,x):
        return np.exp(x)/( 1.0 + np.exp(x))

    def softmax(self,x):
        num = np.exp(x)
        return num/np.sum(num)

    def max_margin(self,scores):
        margin = np.maximum(0,1-self.sigmoid(scores[0]) + self.sigmoid(scores[1:]))
        return np.sum(margin)



    def numpy_s_bilinear_model(self,x_s,X_t,x_r,W):
        n_scores = []
        cost = 0
        for i in range(x_s.shape[0]):
            scores = []
            targets = np.transpose(X_t[i])
            for v in targets:
                outer = np.outer(x_s[i],v)
                z = np.zeros(x_r.shape[1])
                for k in range(x_r.shape[1]):
                    z[k] = np.sum(np.sum(W[k]*outer,axis=0))
                scores.append(np.dot(z,x_r[i]))
            cost += self.max_margin(scores)
            n_scores.append(scores)
        return np.asarray(n_scores),cost

    def test_s_bilinear_model(self):
        n = 4
        dim = 10
        r_dim = 10
        num_negs = 8
        X_s = 0.1 * np.random.normal(size=[n,dim,1])
        X_t = 0.1 * np.random.normal(size=[n,dim,num_negs])
        W = np.random.uniform(size = [dim,dim,dim])
        x_r = np.random.uniform(size=[n,r_dim])
        np_score,np_cost = self.numpy_s_bilinear_model(X_s,X_t,x_r,W)
        #print(np_score)
        s_rescal = get_model("s-rescal")
        score = s_rescal['score']
        fprop = s_rescal['fprop']

        self.assertAlmostEqual(np.linalg.norm(score(X_s,X_t,x_r,W)-np_score),0.0,delta=0.0001)
        self.assertAlmostEqual(fprop(X_s,X_t,x_r,W), np_cost, delta=0.0001)


    def numpy_bilinear_model(self,x_s,x_t,W):
        n_scores = []
        cost = 0.0
        for i in range(x_s.shape[0]):
            scores = []
            X_t = np.transpose(x_t[i])
            for v in X_t:
                scores.append(np.dot(np.dot(W[i],v),np.transpose(x_s[i])))

            cost += self.max_margin(scores)
            n_scores.append(scores)
        return np.asarray(n_scores),cost

    def test_bilinear_model(self):
        n = 4
        dim = 10
        num_negs = 8
        X_s = 0.1*np.random.normal(size=[n,1,dim])
        X_t = 0.1 * np.random.normal(size=[n, dim,num_negs])
        W = np.random.normal(size=[n,dim,dim])
        np_score, np_cost = self.numpy_bilinear_model(X_s,X_t,W)
        bilinear = get_model("bilinear")
        score = bilinear['score']
        fprop = bilinear['fprop']
        self.assertAlmostEqual(np.linalg.norm(score(X_s,X_t,W)-np_score[:,:,0]),0.0,delta=0.0001)
        self.assertAlmostEqual(fprop(X_s,X_t,W), np_cost, delta=0.0001)


    def numpy_transE(self,x_s,X_t,x_r):
        n_scores= []
        cost = 0.0
        for i in range(x_s.shape[0]):
            scores = []
            #targets = np.transpose(X_t[i])
            for t in X_t[i]:
                scores.append(-1.0*np.sum(np.square(x_s[i]+x_r[i]-t)))
            cost += self.max_margin(scores)
            n_scores.append(scores)
        return n_scores,cost

    def test_transE(self):
        dim = 10
        num_negs = 8
        n= 4
        X_s = 0.1 * np.random.normal(size=[n,dim])
        X_t = 0.1 * np.random.normal(size=[n,num_negs,dim])

        x_r = np.random.uniform(size=[n,dim])
        np_score, np_cost = self.numpy_transE(X_s, X_t, x_r)
        transE = get_model('transE')
        score = transE['score']
        fprop = transE['fprop']
        self.assertAlmostEqual(np.linalg.norm(score(X_s, X_t, x_r) - np_score), 0.0, delta=0.0001)
        self.assertAlmostEqual(fprop(X_s, X_t, x_r), np_cost, delta=0.0001)


    def np_tr(self,x_s, x_t,W_r, W_c, pos_cats,neg_cats,alpha):
        cost = 0.0
        for i in range(x_s.shape[0]):
            np_attn, np_pos = self.np_attention(x_s[i], x_t[i], W_r[i], W_c, pos_cats[i])
            neg = self.sigmoid(np.dot(np.dot(W_c,neg_cats[i]).T,x_t[i]))
            margin = 1.0 - self.sigmoid(np_pos) + neg
            pos_margin = np.maximum(np.zeros_like(margin), margin)
            cost += np.sum(pos_margin)
        return cost*alpha

    def test_tr(self):
        dim = 6
        num_cats = 5
        n = 4
        x_s = np.random.randn(n,dim, 1)
        x_t = np.random.randn(n,dim, 1)
        W_r = np.random.randn(n,dim, dim)
        W_c = np.random.randn(dim, dim)
        pos_cats = np.random.randn(n,dim, num_cats)
        neg_cats = np.random.randn(n,dim, num_cats)
        alpha = np.random.randn()
        np_cost = self.np_tr(x_s, x_t,W_r, W_c, pos_cats,neg_cats,alpha)
        tr = get_model('tr')
        cost = tr['fprop'](x_s, x_t,W_r, W_c, pos_cats,neg_cats,alpha)
        self.assertAlmostEqual(cost, np_cost, delta=0.0001)

    '''
    Test Layers
    '''
    def batch_attention(self,x_s, x_t,W_r, W_c, pos_cats):
        scores = 0
        attn = []
        for ind in range(x_s.shape[0]):
            a,score = self.np_attention(x_s[ind],x_t[ind],W_r[ind],W_c,pos_cats[ind])
            scores += score
            attn.append(a[0])
        return np.asarray(attn),scores

    def np_attention(self,x_s, x_t,W_r, W_c, pos_cats):
        a = self.softmax(np.dot(x_s.T,np.dot(np.dot(W_r,W_c),pos_cats)))
        score = np.dot(x_t.T,np.dot(W_c,np.dot(a[0],pos_cats.T)))
        return a, score

    def test_soft_attention(self):
        n= 2
        dim = 6
        num_cats = 5
        x_s = np.random.randn(n,dim, 1)
        x_t = np.random.randn(n,dim, 1)
        W_r = np.random.randn(n,dim, dim)
        W_c = np.random.randn(dim, dim)
        pos_cats = np.random.randn(n,dim, num_cats)
        from theano_models import test_attention
        attn = test_attention()
        np_attn,np_pos = self.batch_attention(x_s, x_t,W_r,W_c,pos_cats)
        attn = attn(x_s, W_r, pos_cats,W_c)
        self.assertAlmostEqual(np.linalg.norm(attn-np_attn),0.0, delta=0.0001)

    def test_max_margin(self):
        scores = np.random.randn(10)
        from theano_models import test_max_margin
        f = test_max_margin()
        self.assertAlmostEqual(self.max_margin(scores),f(scores),delta=0.0001)

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

