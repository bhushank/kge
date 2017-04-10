from data import Path

import util

#W = util.get_correlation_matrix(4)
#print(W)


import models
import numpy as np
import theano
import util
def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))




import time
theano.config.on_unused_input='ignore'
dim = 100
r_dim = 80
X_s = util.to_floatX(np.random.randn(300,dim,1))
x_r = util.to_floatX(np.random.randn(300,r_dim))
X_t = util.to_floatX(np.random.randn(300, dim, 50))
W = util.to_floatX(np.random.randn(r_dim,dim,dim))

theano.config.allow_gc = False
from theano_models import s_rescal
f = s_rescal()
start = time.time()
s = f['fprop'](X_s,X_t,x_r,W,1.0)
#s = f(X_s.T,X_t.T,W,x_r)
end = time.time()
print(end-start)


'''


'''



#w = util.get_correlation_tensor(3)
#print(w)