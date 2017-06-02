__author__ = 'Bhushan Kotnis'

from parameters import SparseParams
import numpy as np
import theano
import util
from theano_models import get_model as get_theano_model
import constants


def get_model(config, neg_sampler):
    model = config['model']

    if model == constants.bilinear:
        return Bilinear(config, neg_sampler)
    elif model == constants.s_rescal:
        return S_Rescal(config, neg_sampler)
    elif model == constants.hole:
        return HolE(config, neg_sampler)
    elif model == constants.transE:
        return TransE(config, neg_sampler)
    else:
        raise NotImplementedError('Model {} not implemented'.format(model))

class Model(object):

    def __init__(self,config,neg_sampler):
        self.model_name = config['model']
        self.e_d = config['entity_dim']
        self.neg_sampler = neg_sampler
        self.neg_samples = constants.num_train_negs
        self.param_scale = config.get('param_scale',constants.param_scale)
        self.model = get_theano_model(self.model_name)
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']
        self.alpha = config.get('alpha',1.0)



    def cost(self,params,batch):
        '''
        Computes the cost function without the regularizer
        :param params: SparseParams
        :param ex: Path
        :return: cost
        '''
        raise NotImplementedError()


    def predict(self,params,ex):
        raise NotImplementedError()

    def gradient(self,params,batch,type_batch_s=None,type_batch_t=None):
        '''
        Computes the gradient SparseParams
        :param params: Initialized SparseParams
        :param ex: Path
        :return grad: SparseParams
        '''
        grad = SparseParams(d=dict())
        # back prop for target negatives
        grad = self.bprop_model(grad, params, batch, True)
        # back prop for source negatives
        grad = self.bprop_model(grad, params, batch, False)

        return grad

    def bprop_model(self,grad,params,ex,is_target):
        raise NotImplementedError()

    def init_f(self,key):
        raise NotImplementedError()


    '''
    Unpacking methods
    '''

    def unpack_batch(self,params,batch):
        x_s_batch = []
        x_t_batch = []
        w_r_batch = []

        for ex in batch:
            x_s, x_t, W_r = self.unpack_triple(params,ex)
            x_s_batch.append(x_s)
            x_t_batch.append(x_t)
            w_r_batch.append(W_r)
        return np.asarray(x_s_batch), np.asarray(x_t_batch), np.asarray(w_r_batch)

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

    def assert_negs(self,is_target,ex,negs):
        if is_target:
            assert ex.t not in negs
        else:
            assert ex.s not in negs

    def unpack_neg_batch(self,params,batch,batch_v,is_target):
        neg_v_batch = []
        negs_batch = []
        for ind,ex in enumerate(batch):
            neg_v,negs = self.get_neg_batch(params,ex,batch_v[ind],is_target)
            neg_v_batch.append(neg_v)
            negs_batch.append(negs)
        return np.asarray(neg_v_batch),negs_batch

    def get_neg_batch(self,params,ex,x,is_target):
        negs = self.neg_sampler.sample(ex, self.neg_samples, is_target)
        if len(negs)>0:
            self.assert_negs(is_target,ex,negs)
            v_negs = self.unpack_entities(params, negs)
            X_batch = np.append(x, v_negs, axis=1)
            return X_batch,negs
        return list(),list()

    '''
    Gradient Collection methods
    '''
    def enforce_shape(self,nd_array):
        '''
        Force entity vector to conform to e_d x 1
        :param nd_array:
        :return:
        '''
        if len(nd_array.shape) <=1:
            return np.reshape(nd_array,(nd_array.shape[0],1))
        elif nd_array.shape[1] > nd_array.shape[0]:
            return np.transpose(nd_array)
        else:
            return nd_array

    def collect_batch_entity_grads(self,grad,entities,g_e,enforce_shape):
        for ind,e in enumerate(entities):
            grad = self.collect_entity_grads(grad,e,g_e[ind],enforce_shape)
        return grad

    def collect_entity_grads(self,grad,e,g_e,enforce_shape):
        return self.add_to_grad(grad,('e',e),enforce_shape(g_e))

    def collect_rel_grads(self,grad,rels,g_r):
        assert isinstance(rels,tuple)
        return self.add_to_grad(grad,('r',rels[0]),g_r)

    def add_to_grad(self,grad,key,val):
        if key in grad:
            grad[key] += val
        else:
            grad[key] = val
        return grad



    def collect_batch_grads(self, grad, batch, gx_s, gx_t, gW_r, negs, is_target):
        for ind, ex in enumerate(batch):
            if is_target:
                grad = self.collect_entity_grads(grad, ex.s, gx_s[ind], self.enforce_shape)
                negs[ind].insert(0, ex.t)
                grad = self.collect_batch_entity_grads(grad, negs[ind], gx_t[ind].T, self.enforce_shape)
            else:
                grad = self.collect_entity_grads(grad, ex.t, gx_t[ind], self.enforce_shape)
                negs[ind].insert(0, ex.s)
                grad = self.collect_batch_entity_grads(grad, negs[ind], gx_s[ind].T, self.enforce_shape)

            grad = self.collect_rel_grads(grad, ex.r, gW_r[ind])

        return grad


class Bilinear(Model):

    def __init__(self,config,neg_sampler):
        super(Bilinear,self).__init__(config,neg_sampler)


    def cost(self,params,batch):
        x_s, x_t, W_r = self.unpack_batch(params,batch)
        X_t_batch,t_negs = self.unpack_neg_batch(params,batch,x_t,True)
        x_s = np.transpose(x_s,axes=[0,2,1])
        return self.fprop(x_s, X_t_batch, W_r)/(len(batch))


    def predict(self,params,batch):
        x_s, x_t, W_r = self.unpack_batch(params, batch)
        x_s = np.transpose(x_s, axes=[0, 2, 1])
        return self.score(x_s, x_t, W_r)


    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, self.e_d))


    def bprop_model(self, grad, params, batch, is_target=True):
        x_s, x_t, W_r = self.unpack_batch(params, batch)

        if is_target:
            X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
            x_s = np.transpose(x_s,axes=[0,2,1])
            gx_s, gx_t, gW_r = self.bprop(x_s, X_t_batch, W_r)
            self.collect_batch_grads(grad,batch,np.transpose(gx_s,axes=[0,2,1]),gx_t,gW_r,negs,is_target)
        else:
            X_s_batch,negs = self.unpack_neg_batch(params,batch,x_s,is_target)
            X_s_batch = np.transpose(X_s_batch,axes=[0,2,1])
            gx_s, gx_t, gW_r = self.bprop(X_s_batch, x_t, W_r)
            self.collect_batch_grads(grad, batch, np.transpose(gx_s,axes=[0,2,1]), gx_t, gW_r, negs,is_target)
        return grad


class S_Rescal(Model):
    def __init__(self,config,neg_sampler):
        super(S_Rescal, self).__init__(config, neg_sampler)
        self.r_d = config.get('relation_dim')

    def unpack_params(self,params,batch):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        W = params[(self.model_name, 'W')]
        return x_s,x_t,x_r,W

    def collect_param_grads(self,grad,param_name,g_p):
        return self.add_to_grad(grad,(self.model_name,param_name),g_p)

    def predict(self,params,batch):
        x_s,x_t,x_r,W = self.unpack_params(params,batch)
        return self.score(x_s, x_t, x_r, W)

    def cost(self,params,batch):
        x_s, x_t, x_r, W = self.unpack_params(params, batch)
        X_t_batch, t_negs = self.unpack_neg_batch(params, batch, x_t, True)
        return self.fprop(x_s, X_t_batch, x_r, W)/(len(batch))

    def bprop_model(self,grad,params,batch,is_target):
        x_s, x_t, x_r, W = self.unpack_params(params, batch)
        if is_target:
            X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
            gx_s, gx_t, gx_r, gW = self.bprop(x_s, X_t_batch, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)
        else:
            X_s_batch, negs = self.unpack_neg_batch(params, batch, x_s, is_target)
            gx_s, gx_t, gx_r, gW = self.bprop(X_s_batch,x_t, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)

        grad = self.collect_param_grads(grad, 'W', gW)
        return grad


    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d))
        if key[1] == 'W':
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d, self.e_d, self.e_d))
        raise NotImplementedError('Param not found')

class HolE(S_Rescal):
    def __init__(self,config,neg_sampler):
        #config_cp = copy.copy(config)
        super(HolE, self).__init__(config, neg_sampler)

    def bprop_model(self,grad,params,batch,is_target):
        x_s, x_t, x_r, W = self.unpack_params(params, batch)
        if is_target:
            X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
            gx_s, gx_t, gx_r = self.bprop(x_s, X_t_batch, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)
        else:
            X_s_batch, negs = self.unpack_neg_batch(params, batch, x_s, is_target)
            gx_s, gx_t, gx_r = self.bprop(X_s_batch, x_t, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)

        return grad

    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d, 1))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d))
        if key[1] == 'W':
            # Correlation tensor is fixed, leads to fixed partitions
            return util.get_correlation_tensor(self.e_d)
        raise NotImplementedError('Param not found')


class TransE(Model):
    def __init__(self,config,neg_sampler):
        super(TransE, self).__init__(config, neg_sampler)

    def cost(self,params,batch):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, True)
        return self.fprop(x_s, np.transpose(X_t_batch, axes=[0, 2, 1]), x_r)/len(batch)


    def predict(self,params,batch):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        return self.score(x_s, np.asarray([x_t]), x_r)

    def init_f(self,key):
        assert isinstance(key, tuple)
        if key[0] == 'e':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d))
        if key[0] == 'r':
            return util.to_floatX(self.param_scale * np.random.randn(self.e_d))


    def bprop_model(self, grad, params, batch,is_target=True):

        x_s, x_t, x_r = self.unpack_batch(params, batch)
        if is_target:
            X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
            gx_s, gx_t, gx_r = self.bprop(x_s,  np.transpose(X_t_batch, axes=[0, 2, 1]), x_r)
            self.collect_batch_grads(grad, batch, gx_s, np.transpose(gx_t, axes=[0, 2, 1]), gx_r, negs, is_target)
        else:
            X_s_batch, negs = self.unpack_neg_batch(params, batch, x_s, is_target)
            gx_t, gx_s, gx_r = self.bprop(x_t, np.transpose(X_s_batch, axes=[0, 2, 1]), x_r )
            self.collect_batch_grads(grad, batch, np.transpose(gx_s,axes=[0, 2, 1]), gx_t, gx_r, negs, is_target)

        return grad

    def enforce_shape(self,nd_array):
        '''
        Force entity vector to conform to e_d x 1
        :param nd_array:
        :return:
        '''
        return nd_array


    def get_neg_batch(self,params,ex,x,is_target):
        negs = self.neg_sampler.sample(ex, self.neg_samples, is_target)
        if len(negs)>0:
            self.assert_negs(is_target, ex, negs)
            v_negs = self.unpack_entities(params, negs)
            if len(negs)==1:
                v_negs = np.reshape(v_negs,(v_negs.shape[0],1))
            X_batch = np.append(np.reshape(x,(x.shape[0],1)), v_negs, axis=1)
            return X_batch,negs
        return list(),list()




