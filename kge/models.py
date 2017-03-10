__author__ = 'Bhushan Kotnis'

from parameters import SparseParams
import numpy as np
import theano
import util
from theano_models import get_model as get_theano_model
from data import load_categories
import constants
import copy

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
    elif model == constants.tr:
        return TypeRegularizer(config,neg_sampler)
    else:
        raise NotImplementedError('Model {} not implemented'.format(model))

class Model(object):

    def __init__(self,config,neg_sampler):
        self.model_name = config['model']
        self.e_d = config['entity_dim']
        self.neg_sampler = neg_sampler
        self.neg_samples = config.get('num_dev_negs',constants.num_dev_negs)
        self.param_scale = config.get('param_scale',constants.param_scale)
        self.model = get_theano_model(self.model_name)
        self.fprop = self.model['fprop']
        self.bprop = self.model['bprop']
        self.score = self.model['score']


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

    def gradient(self,params,batch):
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
            grad = self.collect_entity_grads(grad,('e',e),g_e[ind],enforce_shape)
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
        self.r_d = config.get('relation_dim',config['entity_dim'])

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
            return util.to_floatX(self.param_scale * np.random.randn(self.r_d, self.r_d, self.r_d))
        raise NotImplementedError('Param not found')

class HolE(S_Rescal):
    def __init__(self,config,neg_sampler):
        config_cp = copy.copy(config)
        config_cp['relation_dim'] = config['entity_dim']
        config_cp['model'] = constants.s_rescal
        super(HolE, self).__init__(config_cp, neg_sampler)

    def bprop_model(self,grad,params,batch,is_target):
        x_s, x_t, x_r, W = self.unpack_params(params, batch)
        if is_target:
            X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
            gx_s, gx_t, gx_r, _ = self.bprop(x_s, X_t_batch, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)
        else:
            X_s_batch, negs = self.get_neg_batch(params, batch, x_s, is_target)
            gx_s, gx_t, gx_r, _ = self.bprop(X_s_batch, x_t, x_r, W)
            self.collect_batch_grads(grad, batch, gx_s, gx_t, gx_r, negs, is_target)

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
    def __init__(self,config,neg_sampler):
        super(TransE, self).__init__(config, neg_sampler)

    def cost(self,params,batch):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        X_t_batch, t_negs = self.unpack_neg_batch(params, batch, x_t, True)
        return self.fprop(x_s, X_t_batch.T, x_r)/len(batch)


    def predict(self,params,batch):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        return self.score(x_s, np.asarray([x_t]), x_r)

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

    def bprop_model(self, grad, params, batch,is_target=True):
        x_s, x_t, x_r = self.unpack_batch(params, batch)
        X_t_batch, negs = self.unpack_neg_batch(params, batch, x_t, is_target)
        gx_s, gx_t, gx_r = self.bprop(x_s, np.transpose(X_t_batch,axes=[0,2,1]), x_r)
        self.collect_batch_grads(grad, batch, gx_s, np.transpose(gx_t,axes=[0,2,1]), gx_r, negs, is_target)
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


class TypeRegularizer(Bilinear):
    def __init__(self,config,neg_sampler):
        config_cp = copy.copy(config)
        config_cp['model'] = constants.bilinear
        super(TypeRegularizer, self).__init__(config_cp, neg_sampler)

        self.alpha = util.to_floatX(config['alpha'])
        type_reg = get_theano_model(constants.tr)
        self.tr_fprop = type_reg['fprop']
        self.tr_bprop = type_reg['bprop']
        self.attn = type_reg['attn']

    def unpack_categories(self,params,batch,is_target):
        pos_batch = []
        neg_batch = []
        pos_v_cats_batch = []
        neg_v_cats_batch = []
        for ex in batch:
            if is_target:
                pos_cats = list(self.neg_sampler.sample_pos_cats(ex,is_target))
            else:
                pos_cats = list(self.neg_sampler.sample_pos_cats(ex,is_target))

            neg_cats = self.neg_sampler.sample_neg_cats(ex,is_target)

            pos_v_cats = self.process_cats(params,pos_cats,50)
            neg_v_cats = self.process_cats(params, neg_cats, 100)
            pos_batch.append(pos_cats)
            neg_batch.append(neg_cats)
            pos_v_cats_batch.append(pos_v_cats)
            neg_v_cats_batch.append(neg_v_cats)

        return (np.asarray(pos_v_cats_batch),pos_batch),(np.asarray(neg_v_cats_batch),neg_batch)

    def process_cats(self,params,cats,dim):
        while len(cats)>50:
            cats.pop()
        v_cats = self.unpack_entities(params, cats) if len(cats) > 0 \
            else np.zeros((100, 1), dtype=theano.config.floatX)
        v_cats = util.pad_zeros(v_cats, dim)
        return v_cats


    def attention(self,params,batch):
        x_s, x_t, W_r = self.unpack_batch(params,batch)
        W_c = self.unpack_relations(params, (constants.cat_rel,))
        cats = [self.neg_samples.get_cats(ex.r[0]) for ex in batch]
        v_cats = [self.unpack_entities(params, c) for c in cats]
        attn = self.attn(x_s, W_r, W_c,np.asarray(v_cats))
        return attn,cats

    def cost(self,params,batch):
        cost = 0.0
        x_s, x_t, W_r = self.unpack_batch(params, batch)
        X_t_batch, t_negs = self.unpack_neg_batch(params, batch, x_t, True)
        x_s_t = np.transpose(x_s, axes=[0, 2, 1])
        cost += self.fprop(x_s_t, X_t_batch, W_r)
        pos_cats,neg_cats = self.unpack_categories(params,batch,True)
        W_c = self.unpack_relations(params,(constants.cat_rel,))
        cost += self.tr_fprop(x_s, x_t, W_r, W_c,pos_cats[0],neg_cats[0],self.alpha)
        return cost/len(batch)

    def gradient(self,params,ex):
        grad = SparseParams(d=dict())
        # back prop for target negatives
        grad = self.bprop_model(grad, params, ex, True)
        # back prop for source negatives
        grad = self.bprop_model(grad, params, ex, False)
        # Add type regularized gradient
        grad = self.tr_bprop_model(grad,params,ex,True)
        grad = self.tr_bprop_model(grad, params, ex, False)
        return grad

    def tr_bprop_model(self,grad,params,batch,is_target):
        x_s, x_t, W_r = self.unpack_batch(params, batch)
        W_c = self.unpack_relations(params, (constants.cat_rel,))
        if is_target:
            pos_cats, neg_cats = self.unpack_categories(params, batch,is_target)
            gx_s, gx_t, gx_r, gx_c, g_pos, g_neg = self.tr_bprop(x_s, x_t, W_r,
                                                                 W_c,pos_cats[0],neg_cats[0],self.alpha)

            grad = self.collect_grads(grad,batch,gx_s, gx_t,gx_r,[],self.enforce_shape)
            grad = self.collect_cat_grads(grad,pos_cats[1],g_pos)
            grad = self.collect_cat_grads(grad, neg_cats[1], g_neg)
            # W_r transpose if source
        else:
            W_r = np.transpose(W_r,axes=[0,2,1])
            pos_cats, neg_cats = self.unpack_categories(params, batch, is_target)
            gx_s, gx_t, gx_r, gx_c, g_pos, g_neg = self.tr_bprop(x_t, x_s, W_r,
                                                                 W_c, pos_cats[0], neg_cats[0], self.alpha)
            grad = self.collect_grads(grad, batch, gx_s, gx_t, gx_r, [], self.enforce_shape)
            grad = self.collect_cat_grads(grad, pos_cats[1], g_pos)
            grad = self.collect_cat_grads(grad, neg_cats[1], g_neg)

        return grad

    def collect_grads(self, grad, batch, gx_s, gx_t, gW_r,negs, is_target):
        for ind, ex in enumerate(batch):
            if is_target:
                grad = self.collect_entity_grads(grad, ex.s, gx_s[ind], self.enforce_shape)
            else:
                grad = self.collect_entity_grads(grad, ex.t, gx_t[ind], self.enforce_shape)

            grad = self.collect_rel_grads(grad, ex.r, gW_r[ind])

        return grad

    def collect_cat_grads(self,grad,cats,grad_c):
        for ind,c in enumerate(cats):
            grad = self.collect_batch_entity_grads(grad,c,grad_c[ind].T,self.enforce_shape)
        return grad




class CoupledRescal(Model):
    def __init__(self,config,neg_sampler):
        super(CoupledRescal, self).__init__(config, neg_sampler)
