from data import Path

import util
import random
#W = util.get_correlation_matrix(4)
#print(W)
import cPickle as pickle
import numpy as np

def category():
    base = '/home/kotnis/data/'
    node_names_path = base+'relation_metadata/freebase/node_names.tsv'
    cats = pickle.load(open('entity_cat.cpkl'))
    triples_path = base+'bordes/freebase/train'
    node_names = dict()
    with open(node_names_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts)==2:
                names = node_names.get(parts[0],set())
                names.add(parts[1])
                node_names[parts[0]] = names
    print("Finished reading node names")
    writer = open('category_names','w')
    with open(triples_path) as f:
        for line in f:
            s,r,t = line.strip().split("\t")
            if '.' not in r:
                if s in cats and t in cats:
                    for s_n in node_names.get(s,set()):
                        for t_n in node_names.get(t, set()):
                            write_string = "{},{},{},{},{}".format(cats[s],s_n,r,t_n,cats[t])
                            writer.write(write_string+"\n")



def main():
    file_path = "/home/mitarb/kotnis/Data/bordes/freebase/freebase_bilinear_1/"
    h1 = []
    h3 = []
    with open(file_path+"ranks") as f:
        for line in f:
            line = line.strip()
            h1.append(hits(float(line),1))
            h3.append(hits(float(line), 3))
    print("HITS@1 {}, HITS@3 {}".format(np.mean(h1),np.mean(h3)))

    '''
    p = 0.75
    target = '/home/mitarb/kotnis/Data/bordes/freebase_75/train'
    train = '/home/mitarb/kotnis/Data/bordes/freebase/train'
    writer = open(target,'w')
    with open(train) as f:
        for line in f:
            if random.random()<=p:
                writer.write(line)
    writer.close()
    '''

def hits(rank,k):
    if rank<=k:
        return 1.0
    return 0.0

if __name__=='__main__':
    category()


#w = util.get_correlation_tensor(3)
#print(w)