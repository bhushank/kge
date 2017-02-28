__author__ = 'Bhushan Kotnis'

import theano.tensor as T
import theano
import util
import constants

@util.memoize
def get_model(model):
    if model==constants.bilinear:
        return bilinear()
    elif model=="coupled bilinear":
        return coupled_bilinear()
    elif model==constants.s_rescal:
        return s_rescal()
    elif model == constants.transE:
        return transE()
    elif model == constants.tr:
        return type_regularizer()
    else:
        raise NotImplementedError("Model {} not implemented".format(model))


def s_rescal():
    # Includes negatives, hence matrix
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W  = T.tensor3('W')
    x_r = T.vector('r')

    scores = x_r.dot(X_s.T.dot(W).dot(X_t))
    scores = T.reshape(scores, (1, -1))
    cost = max_margin(scores[0])
    gx_s, gx_t, gx_r, gW = T.grad(cost,wrt=[X_s,X_t,x_r,W])

    print('Compiling s-rescal fprop')
    fprop = theano.function([X_s,X_t,x_r,W],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling s-rescal bprop')
    bprop = theano.function([X_s,X_t,x_r,W],[gx_s,gx_t,gx_r,gW],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling s-rescal score')
    score = theano.function([X_s,X_t,x_r,W],scores[0], name = 'score',mode='FAST_RUN')
    score.trust_input = True

    return {'fprop':fprop,'bprop':bprop,'score':score}


def bilinear():
    '''
    Defines the Bilinear (RESCAL) model in theano
    :return: fprop, bprop, score
    '''
    X_s = T.matrix('X_s')
    X_t = T.matrix('X_t')
    W_r = T.matrix('W_r')

    scores = T.dot(X_s, T.dot(W_r, X_t))
    scores = T.reshape(scores,(1,-1))
    cost = max_margin(scores[0])
    gx_s, gx_t, gW_r = T.grad(cost,wrt=[X_s,X_t,W_r])

    print('Compiling bilinear fprop')
    fprop = theano.function([X_s,X_t,W_r],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling bilinear bprop')
    bprop = theano.function([X_s,X_t,W_r],[gx_s,gx_t,gW_r],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling bilinear score')
    score = theano.function([X_s,X_t,W_r],scores[0], name = 'score',mode='FAST_RUN')
    score.trust_input = True

    return {'fprop':fprop,'bprop':bprop,'score':score}



def transE():
    x_s = T.vector('x_s')
    X_t = T.matrix('x_t')
    x_r = T.vector('r')

    def calc_score(x_s,v,x_r):
        return -1.0*T.sum(T.square(x_s + x_r - v))

    results, updates = theano.scan(lambda v: calc_score(x_s, v, x_r), sequences=[X_t], outputs_info=None)

    cost = max_margin(results)
    gx_s, gx_t, gx_r = T.grad(cost, wrt=[x_s, X_t, x_r])

    print('Compiling transE fprop')
    fprop = theano.function([x_s, X_t, x_r], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling transE bprop')
    bprop = theano.function([x_s, X_t, x_r], [gx_s, gx_t, gx_r], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling transE score')
    score = theano.function([x_s, X_t, x_r], results, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}




def type_regularizer():
    x_s = T.matrix('X_s')
    x_t = T.matrix('X_t')
    W_r = T.matrix('W_r')
    W_c = T.matrix('W_c')
    pos_cats = T.matrix('pos')
    neg_cats = T.matrix('neg')
    attn,pos = soft_attention(x_s,x_t,W_r,W_c,pos_cats)
    # Max Margin loss
    neg = T.nnet.sigmoid(T.dot(x_t, T.dot(W_c, neg_cats)))
    margin = 1.0 - T.nnet.sigmoid(pos) + neg
    pos_margin = T.maximum(T.zeros_like(margin), margin)
    cost = T.sum(pos_margin)
    # Gradient
    gx_s, gx_t, gx_r, gx_c, g_pos, g_neg = T.grad(cost, wrt=[x_s, x_t, W_r, W_c,pos_cats,neg_cats])

    print('Compiling soft type_regularizer fprop')
    fprop = theano.function([x_s, x_t, W_r, W_c,pos_cats,neg_cats], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling soft type_regularizer soft bprop')
    bprop = theano.function([x_s, x_t, W_r, W_c,pos_cats,neg_cats],
                            [gx_s, gx_t, gx_r, gx_c, g_pos, g_neg], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling soft type_regularizer attention')
    attn = theano.function([x_s,x_t,W_r,W_c,pos_cats], attn, name='attention', mode='FAST_RUN')
    attn.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'attn': attn}

def max_margin(scores):
    s = T.nnet.sigmoid(scores)
    margin = 1.0 - s[0] + s[1:]
    pos_margin = T.maximum(T.zeros_like(margin),margin)
    cost = T.sum(pos_margin)
    return cost

def softmax_loss(score,y):
    cost = T.nnet.categorical_crossentropy(score, y)[0]
    return cost


def soft_attention(x_s,x_t,W_r,W_c,pos_cats):
    # attention vector for slecting categories
    a = T.nnet.softmax(x_s.T.dot(W_r).dot(W_c).dot(pos_cats))
    score = x_t.T.dot(W_c).dot(pos_cats).dot(a)
    return a,score

def coupling_layer(X_s,X_t,W_p):
    '''
    Sequential Bilinear Model
    :return: symbolic vector of Bilinear scores
    '''

    results, updates = theano.scan(fn=lambda u, v, W: T.nnet.sigmoid(T.dot(u.T, T.dot(W, v))), outputs_info=None,
                                   sequences=[X_s, X_t, W_p])
    return results


def neural_network_layer(results,W_h,b):
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
    results = coupling_layer(X_s,X_t,W_p)
    h = neural_network_layer(results,W_h,b)
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
    results = coupling_layer(X_s, X_t, W_p)
    fprop = theano.function([X_s,X_t,W_p],results,name='fprop',mode='FAST_RUN')
    fprop.trust_imput=True
    return fprop

def test_nn():
    results = T.vector('results')
    W_h = T.matrix('W_h')
    b = T.scalar('b')
    h = neural_network_layer(results,W_h,b)
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
    scores = T.vector()
    cost = max_margin(scores)
    fprop = theano.function([scores], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True
    return fprop
