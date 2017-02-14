#__author__=Bhushan Kotnis

import theano.tensor as T
import theano
import util

@util.memoize
def get_model(model):
    if model=="bilinear":
        return bilinear()
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
    W  = T.tensor3('W')
    x_r = T.vector('r')


    scores = T.nnet.sigmoid(x_r.dot(X_s.T.dot(W).dot(X_t)))
    cost = max_margin(scores)
    gx_s, gx_t, gW, gx_r = T.grad(cost,wrt=[X_s,X_t,W,x_r])

    print('Compiling Group Interaction fprop')
    fprop = theano.function([X_s,X_t,W,x_r],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling Group Interaction bprop')
    bprop = theano.function([X_s,X_t,W,x_r],[gx_s,gx_t,gW,gx_r],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling Group Interaction Score')
    score = theano.function([X_s,X_t,W,x_r],scores, name = 'score',mode='FAST_RUN')
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

    #results,updates = theano.scan(lambda u,v:T.dot(u.T,T.dot(W_r,v)),sequences=[X_s,X_t],outputs_info=None)
    scores = T.nnet.sigmoid(T.dot(X_s, T.dot(W_r, X_t)))
    cost = max_margin(scores)
    gx_s, gx_t, gW_r = T.grad(cost,wrt=[X_s,X_t,W_r])

    print('Compiling bilinear fprop')
    fprop = theano.function([X_s,X_t,W_r],cost,name='fprop',mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling bilinear bprop')
    bprop = theano.function([X_s,X_t,W_r],[gx_s,gx_t,gW_r],name='bprop',mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling bilinear score')
    score = theano.function([X_s,X_t,W_r],scores, name = 'score',mode='FAST_RUN')
    score.trust_input = True

    return {'fprop':fprop,'bprop':bprop,'score':score}



def holographic_embedding():
    #Same as the group interaction model, but W has a specific pattern and is fixed
    # Includes negatives, hence matrix
    X_s = T.matrix('x_s')
    X_t = T.matrix('x_t')
    W = T.tensor3('W')
    x_r = T.vector('r')

    h = x_r.dot(X_s.T.dot(W).dot(X_t))
    scores = T.nnet.sigmoid(h)
    cost = max_margin(scores)
    gx_s, gx_t, gx_r = T.grad(cost, wrt=[X_s, X_t, x_r], consider_constant=[W])

    print('Compiling hole fprop')
    fprop = theano.function([X_s, X_t, W, x_r], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_imput = True

    print('Compiling hole bprop')
    bprop = theano.function([X_s, X_t, W, x_r], [gx_s, gx_t, gx_r], name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print('Compiling hole score')
    score = theano.function([X_s, X_t, W, x_r], scores, name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'fprop': fprop, 'bprop': bprop, 'score': score}




def transE():
   return None


def ermlp():
    return None


def max_margin(scores):
    margin = 1.0 - scores[0,0] + scores[0,1:]
    pos_margin = T.maximum(T.zeros_like(margin),margin)
    cost = T.sum(pos_margin)
    #negatives = results[1:]
    #cost = (T.maximum(T.zeros_like(negatives), margin -(results[0] + negatives) ))
    return cost

def softmax_cost(score,y):

    # Outer product is a column vector
    cost = T.nnet.categorical_crossentropy(score, y)[0]
    return cost






