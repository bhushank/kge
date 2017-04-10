__author__ = 'Bhushan Kotnis'

import util
import cPickle as pickle
import os
import numpy as np
from collections import defaultdict
from parameters import SparseParams,Initialized
import copy
import constants
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

    def __init__(self,triples):
        self._triples = set(triples)
        self._l_entities, self._entity_set = self._get_entities(triples)
        self.total_ents = copy.copy(len(self._l_entities))
        self.s_filter = self.compute_filter(False)
        self.t_filter = self.compute_filter(True)


    def compute_filter(self,is_target):
        filter = dict()
        for ex in self._triples:
            key = (ex.s,ex.r) if is_target else (ex.r,ex.t)
            candidates = filter.get(key,set())
            if is_target:
                candidates.add(ex.t)
            else:
                candidates.add(ex.s)
            filter[key] = candidates
        return filter

    def _get_entities(self,data):
        entities  = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities), set(entities)

    def sample(self,ex,num_samples,is_target=True):
        samples = set()
        gold = ex.t if is_target else ex.s
        if len(self._entity_set) <= num_samples:
            assert len(self._entity_set) == self.total_ents
            return self._entity_set
        # Number of entities is very high
        while True:
            idx = np.random.randint(0, len(self._l_entities))
            p = Path(ex.s, ex.r, self._l_entities[idx]) if is_target else  Path(self._l_entities[idx], ex.r, ex.t)
            if p not in self._triples and (not self._l_entities[idx] == gold):
                    samples.add(self._l_entities[idx])

            if len(samples) >= num_samples:
                return list(samples)

    def bordes_negs(self, ex, is_target, num_negs=None):

        known_candidates = self.t_filter[(ex.s, ex.r)] if is_target else self.s_filter[(ex.r, ex.t)]
        samples = self._entity_set.copy()
        for e in known_candidates:
            samples.remove(e)
        if num_negs is not None:
            samples = np.random.choice(list(samples), num_negs, replace=False)
        gold = ex.t if is_target else ex.s
        if gold in samples:
            samples.remove(gold)
        return samples


@util.memoize
def load_params(params_path,model):
    print("Loading Params from {}".format(params_path))

    with open(params_path,'r') as f:
        params = pickle.load(f)
    print("Finished Loading Params.")
    return Initialized(SparseParams(params),model.init_f)


def read_dataset(path,dev_mode=True,max_examples = float('inf')):

    data_set = {}
    data_set['train'] = read_file(os.path.join(path,'train'),max_examples)
    if dev_mode:
        data_set['test'] = read_file(os.path.join(path,'dev'),max_examples)
    else:
        data_set['test'] = read_file(os.path.join(path, 'test'), max_examples)
    data_set['dev'] = read_file(os.path.join(path,'dev'),max_examples)

    return data_set

def read_file(f_name,max_examples):
    data = []
    count = 0
    with open(f_name) as f:
        for line in f:
            if count >= max_examples:
                return data
            parts = line.strip().split("\t")
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

