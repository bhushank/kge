from data import Path

import util

#W = util.get_correlation_matrix(4)
#print(W)


import models
import numpy as np

'''
dim = 10
num_negs = 10
X_s = []
X_t = []
for i in range(num_negs):
    X_s.append(0.1*np.random.normal(size=[dim]))
    X_t.append(0.1*np.random.normal(size=[dim]))
W = np.random.normal(size=[dim,dim])
X_s = np.asarray(X_s)
X_t = np.asarray(X_t)
bilinear = models.bilinear()
s = bilinear['score']
c = bilinear['fprop']
scores = s(X_s,X_t,W)
print scores
mm = models.test_max_margin()
sc = np.asarray([ 0.1,  0.01,   0.03,])#  0.5,  0.4,  0.4,0.5,  0.4 , 0.5 , 0.5])
print mm(sc)
print(c(X_s,X_t,W))

def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))
'''
'''
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

import json
data = 'freebase'
model  = 'sbilinear'
base = "/home/mitarb/kotnis/Data/grouped_bilinear/{}/experiment_specs/".format(data)
exp_name = "{}_{}".format(data,model) + "{}.json"
config = json.load(open(base + exp_name.format("")))
l2_arr = [0.001,0.0001,0.01,0.1,0.00001]
for l2 in l2_arr:
    config['l2_reg'] = l2
    json.dump(config,open(base+exp_name.format("_"+str(l2)),'w'))

