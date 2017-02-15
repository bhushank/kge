__author__ = 'Bhushan Kotnis'

import util
import cPickle as pickle
import os
import numpy as np
from collections import defaultdict
from parameters import SparseParams,Initialized

class Path(object):


    def __init__(self, s, r, t,i_ents=None,label=None,aux_rel=None):
        assert isinstance(s, str) and isinstance(t, str)
        assert isinstance(r, tuple)

        if i_ents:
            assert isinstance(i_ents, tuple)
            assert len(i_ents) == len(r)-1
        if label:
            assert isinstance(label,int)
        if aux_rel:
            isinstance(aux_rel,str)

        self.s = s # source
        self.t = t # target
        self.r = r # tuple of relations that makeup a path
        self.label = label # label of the path
        self.i_ents = i_ents
        self.aux_rel = aux_rel # For Coupled Prediction with structural regularizer
        self.scores = []

    def __repr__(self):
        rep = "{} {} {}".format(self.s,self.r,self.t)

        if self.label:
            rep +=" {}".format(self.label)
        return rep

    def __eq__(self, other):
        if not isinstance(other,Path):
            return False
        equal = self.s == other.s and self.t == other.t and self.r == other.r
        if self.i_ents:
            return equal and self.i_ents == other.i_ents
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        hash_p = self.s.__hash__() + self.r.__hash__() + self.t.__hash__()
        if self.i_ents:
            hash_p += self.i_ents.__hash__()
        return hash_p

    def from_triples(triples):
        return [Path(s,(r,),t) for (s,r,t) in triples]


class NegativeSampler(object):

    def __init__(self,train,test,data_path,max_samples=10,is_dev = True,is_train = True):
        self._train = set(train)
        self._test = set(test)
        self._train_negs = self._get_negs(train)
        self._test_negs = self._get_negs(test)
        self.is_dev = is_dev
        self.max_samples = max_samples
        self._is_train = is_train
        self._entities = self._get_entities(train)
        # if this is for final test dataset, then read dev data.
        # Test negatives cannot be dev positives
        if not is_dev:
            dev = read_file(os.path.join(data_path,'dev'),float('inf'))
            self._dev_negs = self._get_negs(dev)
            self._dev = set(dev)

    def _get_entities(self,data):
        entities = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities)

    def _get_negs(self,data):
        negatives = defaultdict(lambda : tuple([set(),set()]))
        for ex in data:
            r = ex.r[0]
            val = negatives[r]
            val[0].add(ex.s)
            val[1].add(ex.t)
            negatives[r] = val

        return negatives

    def _sample(self,entities,ex,is_source,num_samples):
        samples = set()
        num_tries = 0
        #assert ex in self._train
        if len(entities) <= 1:
            return samples
        threshold = len(entities)-1 if len(entities) <= num_samples else 5000
        #rand = np.random.RandomState(seed=3456)
        while True:
            # ToDO: Highly inefficient way to sample

            idx = num_tries if len(entities) <= num_samples else np.random.randint(0, len(entities))
            p = Path(entities[idx], ex.r, ex.t) if is_source else Path(ex.s, ex.r, entities[idx])
            if p not in self._train and entities[idx] not in samples:
                # Correct order, if is_dev then test = dev, else dev neq test
                if not self._is_train:
                    if p not in self._test:
                        if not self.is_dev:
                            if p not in self._dev:
                                samples.add(entities[idx])
                        else:
                            samples.add(entities[idx])
                else:
                    samples.add(entities[idx])

            if len(samples) == min(len(entities)-1,num_samples) or num_tries >= threshold:
                return list(samples)
            num_tries += 1

    def get_samples(self,ex,is_source=False,num_samples=-1):
        # if training data, then get negatives from train_negs
        r = ex.r[0]
        if num_samples < 0:
            num_samples = self.max_samples
        # Typed Negatives for train

        if self._is_train:
            entities = list(self._train_negs[r][0]) if is_source else list(self._train_negs[r][1])
            return self._sample(entities,ex,is_source,num_samples)
        else:
            return self._sample(self._entities, ex, is_source, num_samples)
        '''
        else:
            #could be dev or test
            if self.is_dev:
                # use train_neg and test_neg (dev)
                entities_tr = list(self._train_negs[r][1])
                entities_dev = list(self._test_negs[r][1])
                entities_tr.extend(entities_dev)
                return self._sample(list(set(entities_tr)), ex,is_source,num_samples)
            else:
                # use train_neg, dev_neg and test_neg
                entities_tr = list(self._train_negs[r][1])
                entities_dev = list(self._dev_negs[r][1])
                entities_test = list(self._test_negs[r][1])
                entities_tr.extend(entities_dev)
                entities_tr.extend(entities_test)
                return self._sample(list(set(entities_tr)), ex,is_source, num_samples)
        '''



@util.memoize
def load_params(params_path,init_f):
    print("Loading Params from {}".format(params_path))

    with open(params_path,'r') as f:
        params = pickle.load(f)
    print("Finished Loading Params.")
    return Initialized(SparseParams(params),init_f)


def read_dataset(path,dev_mode=True,max_examples = float('inf'),is_path_data=False):

    data_set = {}
    data_set['train'] = read_file(os.path.join(path,'train'),max_examples,is_path_data)
    if dev_mode:
        data_set['test'] = read_file(os.path.join(path,'dev'),max_examples,is_path_data)
    else:
        data_set['test'] = read_file(os.path.join(path, 'test'), max_examples, is_path_data)

    return data_set

def read_file(f_name,max_examples,is_path_data=False):
    data = []
    count = 0
    with open(f_name) as f:
        for line in f:
            if count >= max_examples:
                return data
            parts = line.strip().split("\t")
            if is_path_data:
                #ToDO: if model is single, then paths need to be converted to triples
                s, t, aux_rel, path, label = parts
                path_parts = path.split('-')
                entities = path_parts[1::2]
                rels = path_parts[::2]
                rels, entities, t = add_blanks(rels,entities)
                p = Path(s, tuple(rels), t, i_ents=tuple(entities), label=label, aux_rel=aux_rel)
                data.append(p)
            else:
                s, r, t = parts
                rels = []
                rels.append(r)
                p = Path(s, tuple(rels), t)
                data.append(p)

            count += 1
    return data

def add_blanks(rels,entities,t):
    if len(rels) == 4:
        return rels,entities,t
    for i in range(len(rels),4):
        rels.append('r_pad')
    entities.append(t)

    for i in range(len(entities),3):
        entities.append('e_pad')

    return rels,entities,'e_pad'




