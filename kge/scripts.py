from data import Path

import util

#W = util.get_correlation_matrix(4)
#print(W)


import models
import numpy as np


def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))


X_s = []
X_t = []
dim = 6
x_s = np.random.randn(dim)
x_r = np.random.randn(dim)
for i in range(3):
    X_t.append(np.random.randn(dim))

X_t = np.asarray(X_t)

from theano_models import transE
f = transE()
s = f['score'](x_s,X_t,x_r)
print(s)



'''
X_s = []
X_t = []
dim = 6
x_s = np.random.randn(dim,1)
W = np.random.randn(dim,dim,dim)
x_r = np.random.randn(dim)
for i in range(3):
    X_t.append(np.random.randn(dim))

X_t = np.transpose(np.asarray(X_t))

from theano_models import s_rescal
f = s_rescal()
s = f['score'](x_s,X_t,x_r,W)
#s = f(X_s.T,X_t.T,W,x_r)
print(s)
print(s.shape)


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
'''
import json
data = 'freebase'
model  = 'transe'
base = "/home/mitarb/kotnis/Data/grouped_bilinear/{}/experiment_specs/".format(data)
exp_name = "{}_{}".format(data,model) + "{}.json"
config = json.load(open(base + exp_name.format("")))
l2_arr = [0.001,0.0001,0.01,0.1,0.00001]
for l2 in l2_arr:
    config['l2_reg'] = l2
    json.dump(config,open(base+exp_name.format("_"+str(l2)),'w'))

'''

#w = util.get_correlation_tensor(3)
#print(w)