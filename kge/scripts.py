from data import Path

import util

#W = util.get_correlation_matrix(4)
#print(W)


import models
import numpy as np
import theano

def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))




import time

dim = 200
X_s = np.random.randn(300,1,dim)
W_r = np.random.randn(300,dim,dim)
X_t = np.random.randn(300,dim,50)

theano.config.allow_gc = False
from theano_models import bilinear
f = bilinear()
start = time.time()
s = f['fprop'](X_s,X_t,W_r)
#s = f(X_s.T,X_t.T,W,x_r)
end = time.time()
print(end-start)


'''

dim = 6
num_cats = 5
x_s = np.random.randn(dim,1)
x_t = np.random.randn(dim,1)
W_r = np.random.randn(dim,dim)
W_c = np.random.randn(dim,dim)
pos_cats = np.random.randn(dim,num_cats)
neg_cats = np.random.randn(dim,num_cats+4)

from theano_models import test_attention, type_regularizer
attn,pos = test_attention()
print(attn(x_s, W_r, W_c, pos_cats))
print(pos(x_s, x_t, W_r, W_c, pos_cats))

f = type_regularizer()
cost = f['fprop']
print(cost(x_s, x_t, W_r, W_c, pos_cats,neg_cats))






score = models.get_model('er-mlp')
dim = 3
X_s = []
X_t = []
x_r = np.random.randn(dim)
np_score = []
C = np.random.randn(dim,3*dim)
w = np.random.randn(1,dim)

for i in range(3):
    X_s.append(np.random.randn(dim))
    X_t.append(np.random.randn(dim))
    x = np.concatenate([X_s[i], X_t[i], x_r])
    np_score.append(sigmoid(np.dot(w,np.dot(C,x))))

s = score['score'](np.asarray(X_s),np.asarray(X_t),x_r,C,w)
print(np_score)
print(s)
'''
''''
paths = set()
p1 = Path('s1',('r1',),'t1')
p2 = Path('s2',('r2',),'t2')
p3 = Path('s3',('r3',),'t3')
print(p1)
paths.add(p1)
paths.add(p2)
paths.add(p3)

p_hit = Path('s1',('r1',),'t1')
p_miss = Path('s1',('r2',),'t1')
print(p_hit in paths)
print(p_miss in paths)
assert p_hit in paths and p_miss not in paths
'''



#w = util.get_correlation_tensor(3)
#print(w)