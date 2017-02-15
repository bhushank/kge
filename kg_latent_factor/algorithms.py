__author__ = 'Bhushan Kotnis'

from parameters import SparseParams
import numpy as np
from models import get_model
import theano
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

class Evaluater(object):

    def evaluate(self,params,x):
        raise NotImplementedError()

class RankEvaluater(Evaluater):
    def __init__(self,objective,neg_sampler,num_samples):
        self.neg_sampler = neg_sampler
        self.objective = objective
        self.num_samples = num_samples
        self.init_score = 10000 # because lower the better
        self.metric_name = "Mean Rank"

    def comparator(self,curr_score,prev_score):
        # write if curr_score less than prev_score
        return curr_score < prev_score

    def evaluate(self,params,ex):
        pos = self.objective.predict(params,ex)
        negs = self.neg_sampler.get_samples(ex)
        if len(negs) < 1:
            return np.nan
        scores = []
        for n in negs:
            ex = Path(ex.s,ex.r,n)
            scores.append(self.objective.predict(params,ex))
        scores.append(pos)
        scores = np.asarray(scores)
        scores = np.reshape(scores,(scores.shape[0],))

        ranks = util.ranks(scores,ascending=False)
        if ranks is None:
            return np.nan
        return ranks[-1]

#ToDO: Use inheritence and implement different versions of cost, gradient and predict for cleaner code
# ToDo: This code is terrible
class Objective(object):
    def __init__(self,model,e_d,r_d,neg_sampler,param_scale=0.1,l2_reg=0.0):
        self.model_name = model
        self.e_d = e_d
        self.r_d = r_d
        if model == 'transE' or model == 'holographic embedding' or model == 'ermlp':
            self.r_d = self.e_d

        self.param_scale = param_scale
        self.model = get_model(model)
        self.neg_sampler = neg_sampler
        # We have enough RAM, disable theano gc
        theano.config.allow_gc = False
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']
        self.l2 = l2_reg
        self.triple_models = {"bilinear", "group interaction","holographic embedding","transE",'er-mlp'}
        self.rel_vector_models = {'holographic embedding','transE','ER-MLP','group interaction','er-mlp'}
        self.param_models = {'holographic embedding','group interaction','er-mlp'}



    def cost(self,params,ex):
        '''
        Computes the cost function without the regularizer
        :param params: SparseParams
        :param ex: Path
        :return: cost
        '''
        cost = 0.0
        # Unpack Params
        x_s, x_t, W_r = self.unpack_triple(params, ex)
        W = None
        w = None
        if self.model_name in self.triple_models:
            s_negs = self.neg_sampler.get_samples(ex, True)
            t_negs = self.neg_sampler.get_samples(ex, False)
            if self.model_name in self.param_models:
                W = self.unpack_params(params,'W')
                w = self.unpack_params(params,'w')
            if len(t_negs)>0:
                # For negative targets
                t_negs = self.unpack_entities(params, t_negs)
                cost += self.fprop_triple_models(x_s,x_t,t_negs,W_r,W,w,is_target = True)
            if len(s_negs) > 0:
                # For negative sources, switch source and targets
                s_negs = self.unpack_entities(params, s_negs)
                cost += self.fprop_triple_models(x_t, x_s, s_negs, W_r,W,w, is_target = False)

        elif self.model_name == 'coupled bilinear':
            W_h,b,w_o,u,c = self.unpack_params(params)
            y = self.unpack_label(ex)
            cost += self.fprop(x_s, x_t, W_r,W_h,b,w_o,u,c,y)

        else:
            raise NotImplementedError()

        return cost

    def gradient(self,params,ex):
        '''
        Computes the gradient SparseParams
        :param params: Initialized SparseParams
        :param ex: Path
        :return grad: SparseParams
        '''
        grad = SparseParams(d=dict())
        # Unpack Params
        x_s, x_t, W_r = self.unpack_triple(params, ex)
        W = None
        w = None
        if self.model_name in self.triple_models:
            s_negs = self.neg_sampler.get_samples(ex,True)
            t_negs = self.neg_sampler.get_samples(ex, False)

            if self.model_name in self.param_models:
                W = self.unpack_params(params,'W')
                w = self.unpack_params(params,'w')

            if len(t_negs) > 0:
                # bprop for negative targets
                t_negs_vectors = self.unpack_entities(params, t_negs)
                self.bprop_triple_models(grad,ex,x_s,x_t,t_negs_vectors,t_negs,W_r,W,w,True)
            if len(s_negs) > 0:
                # bprop for negative sources
                s_negs_vectors = self.unpack_entities(params, s_negs)
                self.bprop_triple_models(grad, ex, x_t, x_s, s_negs_vectors,s_negs, W_r, W, w, False)
            return grad

        elif self.model_name == 'coupled bilinear':
            W_h, b, w_o, u, c = self.unpack_params(params)
            y = self.unpack_label(ex)
            gx_s, gx_t, gW_p, gW_h, g_b, g_w_o, g_u, g_c = self.bprop(x_s, x_t, W_r, W_h, b, w_o, u, c, y)
            self.collect_entity_grads(grad,x_s,list(ex.s) + list(ex.i_ents))
            self.collect_entity_grads(grad, x_t, list(ex.i_ents) + list(ex.t))
            self.collect_rel_grads(grad, ex.r, gW_p)
            self.collect_param_grads(grad,'W_h',gW_h)
            self.collect_param_grads(grad,'b',g_b)
            self.collect_param_grads(grad,'w_o',g_w_o)
            self.collect_param_grads(grad,'u',g_u)
            self.collect_param_grads(grad,'c',g_c)
            return grad
        else:
            raise NotImplementedError()



    def predict(self,params,ex):
        x_s, x_t, W_r = self.unpack_triple(params, ex)


        if self.model_name in self.triple_models:
            x_s = x_s.T
	    x_t = x_t.T
	    #t_negs = self.neg_sampler.get_samples(ex, False)

            #if len(t_negs) > 0:
                # For negative targets
                #negs = self.unpack_entities(params, t_negs)
                #X_t_batch = np.append(x_t, negs, axis=1).T
                # Padding required for source
                #X_s_batch = util.pad(x_s, X_t_batch.shape[0] - 1).T
                #assert X_t_batch.shape == X_s_batch.shape

            if self.model_name == 'bilinear' or self.model_name=='transE':
                return self.score(x_s, x_t, W_r)

            elif self.model_name == 'group interaction' or self.model_name=='holographic embedding':
                W = self.unpack_params(params,'W')
                return self.score(x_s, x_t, W, W_r)
            elif self.model_name == 'er-mlp':
                W = self.unpack_params(params, 'W')
                w = self.unpack_params(params,'w')
                return self.score(x_s, x_t, W_r, W, w)

            else:
                raise NotImplementedError()

        elif self.model_name == 'coupled bilinear':
            W_h, b, w_o, u, c = self.unpack_params(params)
            y = self.unpack_label(ex)
            return self.score(x_s, x_t, W_r, W_h, b, w_o, u, c, y)

        else:
            raise NotImplementedError()

    #ToDO: Refactor, bad and confusing code
    def init_f(self,key):
        assert isinstance(key,tuple)
        rand = np.random.RandomState(seed=12345)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * rand.randn(self.e_d, 1))

        elif self.model_name=='bilinear' or self.model_name == 'coupled bilinear':
            if key[0]=='r':
		#golorot = np.sqrt(2.0/(200))
                return util.to_floatX(self.param_scale*rand.randn(self.e_d,self.e_d))

        elif self.model_name in self.rel_vector_models:
            if key[0] == 'r':
                return util.to_floatX(self.param_scale*rand.randn(self.r_d))
            if key[1] == 'W':
                if self.model_name=='group interaction':
                    return util.get_correlation_matrix(self.e_d)
		    #return util.to_floatX(self.param_scale*rand.randn(self.r_d,self.e_d*self.e_d))
                elif self.model_name == 'holographic embedding':
                    return util.get_correlation_matrix(self.e_d)
                elif self.model_name == 'er-mlp':
                    return util.to_floatX(self.param_scale*rand.randn(self.e_d,3*self.e_d))
                else:
                    return NotImplementedError()
            if key[1] == 'w':
                return util.to_floatX(self.param_scale * rand.randn(1,self.e_d))

        elif self.model_name == 'coupled bilinear':
            if key[1] == 'W_h':
                golorot = np.sqrt(6.0 / 10.0)
                return util.to_floatX(np.random.uniform(low = -golorot,high=golorot,size=(5,5)))
            elif key[1] == 'b' or key[1] == 'c':
                return util.to_floatX(self.param_scale*np.random.randn())
            elif key[1] == 'w_o':
                return util.to_floatX(self.param_scale*np.random.randn(5))
            elif key[1] == 'u':
                # means should be ordered
                u = np.ones((1,5),dtype=theano.config.floatX)
                self.fix_means(u)
                for i in range(1,4):
                    u[0,i] = u[0,i-1] + 0.5
                return u
            else:
                raise KeyError()
        else:
            raise NotImplementedError()



    def bprop_triple_models(self,grad,ex,x_s,x_t,negs,negs_names,W_r,W,w,is_target=True):
        ''''
        Computes gradient for models that accept triples. Switch source, targets for both sides
        '''
        X_t_batch = np.append(x_t, negs, axis=1)
        X_t_batch = X_t_batch.T
        # Padding required for source

        X_s_batch = util.pad(x_s, X_t_batch.shape[0] - 1)
        X_s_batch = X_s_batch.T
        assert X_t_batch.shape == X_s_batch.shape
        # First one is correct, appended are incorrect
        #y = np.zeros((1, X_s_batch.shape[0]),dtype=theano.config.floatX)
        #y[0, 0] = 1.0
        if self.model_name == 'bilinear' or self.model_name=='transE':
            if is_target:
                gx_s, gx_t, gW_r = self.bprop(X_s_batch, X_t_batch, W_r)
                self.collect_entity_grads(grad, [ex.s]*(len(negs_names)+1), gx_s)
                negs_names.insert(0, ex.t)
                self.collect_entity_grads(grad, negs_names, gx_t)
            else:
                gx_s, gx_t, gW_r = self.bprop(X_t_batch, X_s_batch, W_r)
                self.collect_entity_grads(grad, [ex.t] * (len(negs_names) + 1), gx_t)
                negs_names.insert(0, ex.s)
                self.collect_entity_grads(grad, negs_names, gx_s)


        elif self.model_name in self.param_models:
            if is_target:
                if self.model_name=='holographic embedding':
                    gx_s, gx_t, gW_r = self.bprop(X_s_batch, X_t_batch, W, W_r)
                elif self.model_name=='group interaction':
                    gx_s, gx_t, gW, gW_r = self.bprop(X_s_batch, X_t_batch, W, W_r)
                    self.collect_param_grads(grad, 'W', gW)
                elif self.model_name == 'er-mlp':
                    gx_s, gx_t, gW_r, gW, gw = self.bprop(X_s_batch, X_t_batch,W_r, W, w)
                    self.collect_param_grads(grad, 'W', gW)
                    self.collect_param_grads(grad, 'w', gw)

                self.collect_entity_grads(grad, [ex.s] * (len(negs_names) + 1), gx_s)
                negs_names.insert(0, ex.t)
                self.collect_entity_grads(grad, negs_names, gx_t)
            else:
                if self.model_name == 'holographic embedding':
                    gx_s, gx_t, gW_r = self.bprop(X_t_batch, X_s_batch, W, W_r)
                elif self.model_name=='group interaction':
                    gx_s, gx_t, gW, gW_r = self.bprop(X_t_batch, X_s_batch, W, W_r)
                    self.collect_param_grads(grad, 'W', gW)
                elif self.model_name == 'er-mlp':
                    gx_s, gx_t, gW_r, gW, gw = self.bprop(X_t_batch, X_s_batch, W_r, W, w)
                    self.collect_param_grads(grad, 'W', gW)

                self.collect_entity_grads(grad, [ex.t] * (len(negs_names) + 1), gx_t)
                negs_names.insert(0, ex.s)
                self.collect_entity_grads(grad, negs_names, gx_s)

        else:
            raise NotImplementedError()

        self.collect_rel_grads(grad, ex.r, gW_r)

    def fprop_triple_models(self,x_s,x_t,negs, W_r,W,w,is_target=True):
        '''
        Computes cost for models that accept triples. Switch source, targets for both sides
        :param x_s:
        :param x_t:
        :param negs:
        :param W_r:
        :param W:
        :return:
        '''
        X_t_batch = np.append(x_t, negs, axis=1).T
        # Padding required for source
        X_s_batch = util.pad(x_s,X_t_batch.shape[0] - 1 ).T
        assert X_t_batch.shape == X_s_batch.shape
        # First one is correct, appended are incorrect
        #y = np.zeros((1, X_s_batch.shape[0]),dtype=theano.config.floatX)
        #y[0, 0] = 1.0
        if self.model_name == 'bilinear' or self.model_name == 'transE':
            if is_target:
                return self.fprop(X_s_batch, X_t_batch, W_r)
            else:
                return self.fprop(X_t_batch,X_s_batch, W_r)
        elif self.model_name=='group interaction' or self.model_name == 'holographic embedding':
            if is_target:
                return self.fprop(X_s_batch, X_t_batch, W, W_r)
            else:
                return self.fprop(X_t_batch, X_s_batch, W, W_r)
        elif self.model_name == 'er-mlp':
            if is_target:
                return self.fprop(X_s_batch, X_t_batch, W_r, W, w)
            else:
                return self.fprop(X_t_batch, X_s_batch, W_r, W, w)

        else:
            raise NotImplementedError()


    def collect_entity_grads(self,grad,entities,g_e):
        if isinstance(entities,list):
            for i,e in enumerate(entities):
                # enforce num X dim
                if g_e.shape[0] == self.e_d:
                    g_e = np.transpose(g_e)
                self.add_to_grad(grad,('e',e),np.reshape(g_e[i,:],(g_e.shape[1],1)))
        else:
            if g_e.shape[1] == self.e_d:
                g_e = np.transpose(g_e)
            # always dim X 1
            self.add_to_grad(grad,('e',entities),g_e)

    def collect_rel_grads(self,grad,rels,g_r):
        assert isinstance(rels,tuple)
        if len(rels)==1:
            self.add_to_grad(grad,('r',rels[0]),g_r)
        else:
            # Handle 3d tensor
            if self.model_name == 'bilinear' or self.model_name == 'coupled bilinear':
                for i,r in enumerate(rels):
                    self.add_to_grad(grad,('r',r),g_r[i])
            elif self.model_name in self.rel_vector_models:
                for i,r in enumerate(rels):
                    self.add_to_grad(grad,('r',r),g_r[:,i])


    def collect_param_grads(self,grad,param_name,g_p):
        if self.model_name == 'holographic embedding':
            return
        self.add_to_grad(grad,(self.model_name,param_name),g_p)
        if param_name=='u' and self.model_name=='coupled bilinear':
            self.fix_means(grad[(self.model_name,param_name)])



    def add_to_grad(self,grad,key,val):
        if key in grad:
            grad[key] += val
        else:
            grad[key] = val

    def unpack_triple(self,params,ex):

        W_r = self.unpack_relations(params,ex.r)
        x_s = self.unpack_entities(params,ex.s)
        x_t = self.unpack_entities(params,ex.t)
        if self.model_name in self.triple_models:
            return x_s, x_t, W_r
        else:
            #sources s    e1  e2    e3
            #targets e1   e2  e3    t
            X_s = np.append(x_s,self.unpack_entities(params,ex.i_ents))
            X_t = np.append(self.unpack_entities(params,ex.i_ents),x_t)
            return X_s, W_r, X_t

    def unpack_params(self,params,param_name):
        #if self.model_name == "group interaction" or self.model_name == "holographic embedding":
        '''
        elif self.model_name== "coupled bilinear":
            W_h = params[(self.model_name, 'W_h')]
            b = params[(self.model_name, 'b')]
            w_o = params[(self.model_name, 'w_o')]
            u = params[(self.model_name, 'u')]
            c = params[(self.model_name, 'c')]
            #Fix the means
            self.fix_means(u)
            return W_h, b, w_o, u, c

        else:
            raise NotImplementedError()
        '''
        return params[(self.model_name,param_name)]


    def unpack_label(self,ex):
        #ToDo: Make a Constants file and import from there instead of hardcoding
        label = ex.label
        y = np.zeros((1,5),dtype=theano.config.floatX)
        y[0,int(label)] = 1.0
        return y

    def unpack_entities(self,params,entities):
        if isinstance(entities,str):
            return self.build_entity(params,entities)
        else:
            if len(entities) == 1:
                return self.build_entity(params, entities[0])
            return np.asarray([self.build_entity(params,e) for e in entities],dtype=theano.config.floatX).squeeze().T

    def unpack_relations(self,params,rels):
        # triple models expect a vector or matrix
        if len(rels)==1:
            return self.build_relation(params,rels[0])
        # coupled models expect a 3D Tensor
        return np.asarray([self.build_relation(params,r) for r in rels],dtype=theano.config.floatX)

    def build_entity(self,params,e):
        return params[('e',e)]

    def build_relation(self,params,r):
        #handle inverses
        if r.startswith('_'):
            if  self.model_name=='bilinear' or self.model_name=='coupled bilinear':
                return params[('r',r[1:])].T
            else:
                return params[('r', r)]
        return params[('r', r)]

    def fix_means(self,u):
        u[0, 0] = -1
        u[0, -1] = 1
