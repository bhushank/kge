__author__ = 'Bhushan Kotnis'

import theano.tensor as T
import theano
import util

@util.memoize
def get_model(model):
    if model=="bilinear":
        return bilinear()
    elif model=="coupled_bilinear":
        return coupled_bilinear()
    elif model=='group interaction':
        return group_interaction_model()
    elif model == 'holographic embedding':
        return holographic_embedding()
    elif model == 'er-mlp':
        return ermlp()
    elif model == 'transE':
        return transE()
    else:
        raise NotImplementedError()




def group_interaction_model():
    # Includes negatives, hence matrix
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W  = T.matrix('W')
    x_r = T.vector('r')


    def calc_score(u,v,W,x_r):
        outer = T.reshape(T.outer(u,v),(1,-1))[0,:]
        return T.nnet.sigmoid(T.dot(x_r,T.dot(W,outer)))

    results,updates = theano.scan(lambda u,v:calc_score(u,v,W,x_r),sequences=[X_s,X_t],outputs_info=None)
    cost = max_margin(results)
    #score = T.nnet.softmax(results)
    # Outer product is a column vector
    #cost = T.nnet.categorical_crossentropy(score,y)[0]

    gx_s, gx_t, gW, gx_r = T.grad(cost,wrt=[X_s,X_t,W,x_r])

    print('Compiling Group Interaction fprop')
    fprop = theano.function([X_s,X_t,W,x_r],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling Group Interaction bprop')
    bprop = theano.function([X_s,X_t,W,x_r],[gx_s,gx_t,gW,gx_r],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling Group Interaction Score')
    score = theano.function([X_s,X_t,W,x_r],results, name = 'score',mode='FAST_RUN')
    score.trust_input = True

    return {'fprop':fprop,'bprop':bprop,'score':score}


def test():
    X_s = T.matrix('X_s')
    X_t = T.matrix('X_t')
    W_r = T.matrix('W_r')

    results, updates = theano.scan(lambda u, v: T.nnet.sigmoid(T.dot(u.T, T.dot(W_r, v))), sequences=[X_s, X_t],
                                   outputs_info=None)

    fprop = theano.function([X_s, X_t, W_r], results, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True
    return fprop

def bilinear():

    X_s = T.matrix('X_s')
    X_t = T.matrix('X_t')
    W_r = T.matrix('W_r')

    results,updates = theano.scan(lambda u,v:T.nnet.sigmoid(T.dot(u.T,T.dot(W_r,v))),sequences=[X_s,X_t],
                                  outputs_info=None)
    cost = max_margin(results)
    gx_s, gx_t, gW_r = T.grad(cost,wrt=[X_s,X_t,W_r])

    print('Compiling Bilinear fprop')
    fprop = theano.function([X_s,X_t,W_r],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling Bilinear bprop')
    bprop = theano.function([X_s,X_t,W_r],[gx_s,gx_t,gW_r],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling score')
    score = theano.function([X_s,X_t,W_r],results, name = 'score',mode='FAST_RUN')
    score.trust_input = True

    return {'fprop':fprop,'bprop':bprop,'score':score}


def holographic_embedding():
    #Same as the group interaction model, but W has a specific pattern and is fixed
    # Includes negatives, hence matrix
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W = T.matrix('W')
    x_r = T.vector('r')

    def calc_score(u, v, W, x_r):
        outer = T.reshape(T.outer(u, v), (1, -1))[0, :]
        return T.nnet.sigmoid(T.dot(x_r, T.dot(W, outer)))

    results, updates = theano.scan(lambda u, v: calc_score(u, v, W, x_r), sequences=[X_s, X_t], outputs_info=None)
    cost = max_margin(results)

    # score = T.nnet.softmax(results)
    # Outer product is a column vector
    # cost = T.nnet.categorical_crossentropy(score,y)[0]

    gx_s, gx_t, gx_r = T.grad(cost, wrt=[X_s, X_t, x_r], consider_constant=[W])

    print('Compiling hole fprop')
    fprop = theano.function([X_s, X_t, W, x_r], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling hole bprop')
    bprop = theano.function([X_s, X_t, W, x_r], [gx_s, gx_t, gx_r], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling hole score')
    score = theano.function([X_s, X_t, W, x_r], results, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}




def transE():
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    x_r = T.vector('r')
    #y = T.matrix('y')

    def calc_score(u,v,x_r):
        return T.sum(u - (x_r - v))**2

    results, updates = theano.scan(lambda u, v: calc_score(u, v, x_r), sequences=[X_s, X_t], outputs_info=None)
    cost = max_margin(results)
    #score = T.nnet.softmax(results)
    # Outer product is a column vector
    #cost = T.nnet.categorical_crossentropy(score, y)[0]
    gx_s, gx_t, gx_r = T.grad(cost, wrt=[X_s, X_t, x_r])

    print('Compiling trans-e fprop')
    fprop = theano.function([X_s, X_t, x_r], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling trans-e bprop')
    bprop = theano.function([X_s, X_t, x_r], [gx_s, gx_t, gx_r], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling trans-e Score')
    score = theano.function([X_s, X_t, x_r], results, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}


def ermlp():
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    x_r = T.vector('r')
    C = T.matrix('C')
    w = T.matrix('w')


    def calc_score(u,v,x_r,C,w):
        x = T.concatenate([u,v,x_r])
        return T.nnet.sigmoid(T.dot(w,T.dot(C,x)))

    results, updates = theano.scan(lambda u, v: calc_score(u, v, x_r,C,w), sequences=[X_s, X_t], outputs_info=None)
    cost = max_margin(results)

    gx_s, gx_t, gx_r,g_C,g_w = T.grad(cost, wrt=[X_s, X_t, x_r,C,w])

    print('Compiling er-mlp fprop')
    fprop = theano.function([X_s, X_t, x_r, C, w], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling er-mlp bprop')
    bprop = theano.function([X_s, X_t, x_r, C, w], [gx_s, gx_t, gx_r, g_C, g_w], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling er-mlp Score')
    score = theano.function([X_s, X_t, x_r, C, w], results, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}



def max_margin(results):
    margin = 1.0 - results[0] + results[1:]
    pos_margin = T.maximum(T.zeros_like(margin),margin)
    cost = T.sum(pos_margin)
    #negatives = results[1:]
    #cost = (T.maximum(T.zeros_like(negatives), margin -(results[0] + negatives) ))
    return cost


def coupling(X_s,X_t,W_p):
    '''
    Sequential Bilinear Model
    :return: symbolic vector of Bilinear scores
    '''

    results, updates = theano.scan(fn=lambda u, v, W: T.nnet.sigmoid(T.dot(u.T, T.dot(W, v))), outputs_info=None,
                                   sequences=[X_s, X_t, W_p])
    return results


def neural_network(results,W_h,b):
    # matrix (1 X K) for means. u_1 = -1, u_K = 1
    h = T.nnet.sigmoid(T.dot(W_h,results) + b)
    return h

def ordistic_loss(h,u,w_o,c,y):
    z = T.dot(w_o,h) + c
    q = T.exp(u*z + (y - T.sqr(u)/2))
    score = q/q.sum()
    cost = T.nnet.categorical_crossentropy(score,y)
    return score,cost

def coupled_bilinear():
    '''
    Defines the coupled Bilinear (RESCAL) model in theano
    :return: fprop, bprop, score
    '''
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W_p = T.tensor3('W_p')
    y = T.matrix('y')
    # Now define a MLP to combine the scores
    W_h = T.matrix('W_h')
    b = T.scalar('b')
    # Only one vector enforces means to lie on a line. No need to regularize w_o
    w_o = T.vector('w_o')
    # matrix (1 X K) for means. u_1 = -1, u_K = 1
    u = T.matrix('u')
    c = T.scalar('c')

    # The model
    results = coupling(X_s,X_t,W_p)
    h = neural_network(results,W_h,b)
    score,cost = ordistic_loss(h,u,w_o,c,y)
    # Gradient
    gx_s, gx_t, gW_p, gW_h, g_b, g_w_o, g_u,g_c = T.grad(cost, wrt=[X_s, X_t, W_p, W_h, b, w_o, u, c], consider_constant=[y])

    print('Compiling Coupled Bilinear fprop')
    fprop = theano.function([X_s, X_t, W_p,W_h,b,w_o,u,c,y], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling Coupled Bilinear bprop')
    bprop = theano.function([X_s, X_t, W_p, W_h, b, w_o, u, c, y], [gx_s, gx_t, gW_p, gW_h, g_b, g_w_o, g_u, g_c], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling Coupled Bilinear Score')
    score = theano.function([X_s, X_t, W_p, W_h, b, w_o, u, c], score, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}


'''
Tests for layers
'''

def test_coupling():
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W_p = T.tensor3('W_p')
    results = coupling(X_s, X_t, W_p)
    fprop = theano.function([X_s,X_t,W_p],results,name='fprop',mode='FAST_RUN')
    fprop.trust_imput=True
    return fprop

def test_nn():
    results = T.vector('results')
    W_h = T.matrix('W_h')
    b = T.scalar('b')
    h = neural_network(results,W_h,b)
    fprop = theano.function([results,W_h,b],h,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True
    return fprop

def test_ordistic():
    h = T.vector('h')
    w_o = T.vector('w_o')
    # matrix (1 X K) for means. u_1 = -1, u_K = 1
    u = T.matrix('u')
    c = T.scalar('c')
    y = T.matrix('y')
    score = ordistic_loss(h,u,w_o,c,y)
    fprop = theano.function([h,w_o,u,c,y],score,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True
    return fprop


def test_max_margin():
    x = T.vector()
    cost = max_margin(x)
    fprop = theano.function([x],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput=True
    return fprop
