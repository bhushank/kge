__author__ = 'Bhushan Kotnis'

from parameters import SparseParams
import numpy as np
import theano
import util
import theano_models




class Model(object):

    def __init__(self,model,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        self.model_name = model
        self.e_d = e_d
        self.neg_sampler = neg_sampler
        self.num_negs = num_negs
        self.param_scale = param_scale
        self.l2_reg = l2_reg
        self.model = theano_models.get_model(model)
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']


    def cost(self,params,ex):
        '''
        Computes the cost function without the regularizer
        :param params: SparseParams
        :param ex: Path
        :return: cost
        '''
        raise NotImplementedError()


    def predict(self,params,ex):
        raise NotImplementedError()

    def gradient(self,params,ex):
        '''
        Computes the gradient SparseParams
        :param params: Initialized SparseParams
        :param ex: Path
        :return grad: SparseParams
        '''
        grad = SparseParams(d=dict())
        # back prop for target negatives
        grad = self.bprop_model(grad, params, ex, True)
        # back prop for source negatives
        grad = self.bprop_model(grad, params, ex, False)
        return grad

    def bprop_model(self,grad,params,ex,is_target):
        raise NotImplementedError()

    def init_f(self,key):
        raise NotImplementedError()


    '''
    Unpacking methods
    '''
    def unpack_triple(self,params,ex):

        W_r = self.unpack_relations(params,ex.r)
        x_s = self.unpack_entities(params,ex.s)
        x_t = self.unpack_entities(params,ex.t)
        return x_s, x_t, W_r

    def unpack_entities(self,params,entities):
        if isinstance(entities,str):
            return self.build_entity(params,entities)
        else:
            if len(entities) == 1:
                return self.build_entity(params, entities[0])
            return np.asarray([self.build_entity(params,e) for e in entities],dtype=theano.config.floatX).squeeze().T

    def unpack_relations(self,params,rels):
        return self.build_relation(params,rels[0])


    def build_entity(self,params,e):
        return params[('e',e)]

    def build_relation(self,params,r):
        return params[('r', r)]

    def get_neg_batch(self,params,ex,x,is_target):
        t_negs = self.neg_sampler.get_samples(ex, is_target)
        if len(t_negs)>0:
            t_negs = self.unpack_entities(params, t_negs)
            X_t_batch = np.append(x, t_negs, axis=1)
            return X_t_batch,t_negs
        return list(),list()

    '''
    Gradient Collection methods
    '''

    def collect_entity_grads(self,grad,e,g_e):
        return self.add_to_grad(grad,('e',e),g_e)

    def collect_rel_grads(self,grad,rels,g_r):
        assert isinstance(rels,tuple)
        return self.add_to_grad(grad,('r',rels[0]),g_r)

    def add_to_grad(self,grad,key,val):
        if key in grad:
            grad[key] += val
        else:
            grad[key] = val
        return grad



class Bilinear(Model):

    def __init__(self,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        super(Bilinear,self).__init__('bilinear',e_d,neg_sampler,num_negs,param_scale,l2_reg)


    def cost(self,params,ex):
        x_s, x_t, W_r = self.unpack_triple(params, ex)
        X_t_batch,t_negs = self.neg_sampler.get_samples(ex,x_t, False)
        if len(t_negs)>0:
            x_s = np.transpose(x_s)
            return self.fprop(x_s, X_t_batch, W_r)
        return 0.0

    def predict(self,params,ex):
        x_s, x_t, W_r = self.unpack_triple(params, ex)
        x_s = np.transpose(x_s)
        return self.score(x_s, x_t, W_r)


    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, self.e_d))



    def bprop_model(self, grad, params, ex, is_target=True):
        x_s, x_t, W_r = self.unpack_triple(params, ex)

        if is_target:
            X_t_batch, negs = self.get_neg_batch(params,ex,x_t,is_target)
            x_s = np.transpose(x_s)
            gx_s, gx_t, gW_r = self.bprop(x_s, X_t_batch, W_r)
            grad = self.collect_entity_grads(grad, ex.s, gx_s)
            negs.insert(0, ex.t)
            grad = self.collect_entity_grads(grad, negs, gx_t)
        else:
            X_s_batch,negs = self.get_neg_batch(params,ex,x_s,is_target)
            X_s_batch = np.transpose(X_s_batch)
            gx_s, gx_t, gW_r = self.bprop(X_s_batch, x_t, W_r)
            grad = self.collect_entity_grads(grad, ex.t, gx_t)
            negs.insert(0, ex.s)
            grad = self.collect_entity_grads(grad, negs, gx_s)

        grad = self.collect_rel_grads(grad, ex.r, gW_r)
        return grad



class S_Rescal(Model):
    def __init__(self,e_d,r_d,neg_sampler,num_negs,param_scale,l2_reg):
        super(S_Rescal, self).__init__(self.model_name, e_d, neg_sampler, num_negs, param_scale, l2_reg)
        self.r_d = r_d

    def unpack_params(self,params,ex):
        x_s, x_t, x_r = self.unpack_triple(params, ex)
        W = params[(self.model_name, 'W')]
        return x_s,x_t,x_r,W

    def collect_param_grads(self,grad,param_name,g_p):
        return self.add_to_grad(grad,(self.model_name,param_name),g_p)

    def predict(self,params,ex):
        x_s,x_t,x_r,W = self.unpack_params(params,ex)
        return self.score(x_s, x_t, W, x_r)

    def cost(self,params,ex):
        x_s, x_t, x_r, W = self.unpack_params(params, ex)
        t_negs = self.neg_sampler.get_samples(ex, False)
        if len(t_negs) > 0:
            t_negs = self.unpack_entities(params, t_negs)
            X_t_batch = np.append(x_t, t_negs, axis=1)
            return self.fprop(x_s, X_t_batch, W, x_r)
        else:
            return 0.0

    def bprop_model(self,grad,params,ex,is_target):
        x_s, x_t, x_r, W = self.unpack_params(params, ex)
        if is_target:
            X_t_batch, negs = self.get_neg_batch(params, ex, x_t, is_target)
            gx_s, gx_t, gx_r, gW = self.bprop(x_s, X_t_batch, x_r, W)
            grad = self.collect_entity_grads(grad, ex.s, gx_s)
            negs.insert(0, ex.t)
            grad = self.collect_entity_grads(grad, negs, gx_t)
        else:
            X_s_batch, negs = self.get_neg_batch(params, ex, x_s, is_target)
            gx_s, gx_t, gx_r, gW = self.bprop(X_s_batch,x_t, x_r, W)
            grad = self.collect_entity_grads(grad, ex.t, gx_t)
            negs.insert(0, ex.s)
            grad = self.collect_entity_grads(grad, negs, gx_s)

        grad = self.collect_param_grads(grad, 'W', gW)
        grad = self.collect_rel_grads(grad, ex.r, gx_r)
        return grad

    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d))
        if key[1] == 'W':
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d, self.r_d, self.r_d))
        raise NotImplementedError('Param not found')

class HolE(S_Rescal):
    def __init__(self,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        super(HolE, self).__init__(e_d,e_d, neg_sampler, num_negs, param_scale, l2_reg)

    def bprop_model(self,grad,params,ex,is_target):
        x_s, x_t, x_r, W = self.unpack_params(params, ex)
        if is_target:
            X_t_batch, negs = self.get_neg_batch(params, ex, x_t, is_target)
            # Don't update W for HolE
            gx_s, gx_t, gx_r, _ = self.bprop(x_s, X_t_batch, x_r, W)
            grad = self.collect_entity_grads(grad, ex.s, gx_s)
            negs.insert(0, ex.t)
            grad = self.collect_entity_grads(grad, negs, gx_t)
        else:
            X_s_batch, negs = self.get_neg_batch(params, ex, x_s, is_target)
            # Don't update W for HolE
            gx_s, gx_t, gx_r, _ = self.bprop(X_s_batch,x_t, x_r, W)
            grad = self.collect_entity_grads(grad, ex.t, gx_t)
            negs.insert(0, ex.s)
            grad = self.collect_entity_grads(grad, negs, gx_s)

        grad = self.collect_rel_grads(grad, ex.r, gx_r)
        return grad

    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d))
        if key[1] == 'W':
            # Correlation tensor is fixed, leads to fixed partitions
            return util.get_correlation_tensor(self.e_d)
        raise NotImplementedError('Param not found')


class TransE(Model):
    def __init__(self,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        super(TransE, self).__init__('s-rescal', e_d, neg_sampler, num_negs, param_scale, l2_reg)

    def cost(self,params,ex):
        x_s, x_t, x_r = self.unpack_triple(params, ex)
        X_t_batch,t_negs = self.neg_sampler.get_samples(ex,x_t, False)
        if len(t_negs)>0:
            return self.fprop(x_s, X_t_batch, x_r)
        return 0.0

    def predict(self,params,ex):
        x_s, x_t, x_r = self.unpack_triple(params, ex)
        return self.score(x_s, x_t, x_r)

    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d))


    def gradient(self,params,ex):
        grad = SparseParams(d=dict())
        # back prop only for target negatives
        grad = self.bprop_model(grad, params, ex)
        return grad

    def bprop_model(self, grad, params, ex,is_target=True):
        x_s, x_t, x_r = self.unpack_triple(params, ex)
        X_t_batch, negs = self.get_neg_batch(params,ex,x_t,is_target)
        x_s = np.transpose(x_s)
        gx_s, gx_t, gx_r = self.bprop(x_s, X_t_batch, x_r)
        grad = self.collect_entity_grads(grad, ex.s, gx_s)
        negs.insert(0, ex.t)
        grad = self.collect_entity_grads(grad, negs, gx_t)
        grad = self.collect_rel_grads(grad, ex.r, gx_r)
        return grad



class TypeRegularizer(Model):
    def __init__(self,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        self.e_d = e_d
        self.neg_sampler = neg_sampler
        self.num_negs = num_negs
        self.param_scale = param_scale
        self.l2_reg = l2_reg
        self.model = theano_models.bilinear()
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']

class CoupledRescal(Model):
    def __init__(self,e_d,neg_sampler,num_negs,param_scale,l2_reg):
        self.e_d = e_d
        self.neg_sampler = neg_sampler
        self.num_negs = num_negs
        self.param_scale = param_scale
        self.l2_reg = l2_reg
        self.model = theano_models.bilinear()
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']
