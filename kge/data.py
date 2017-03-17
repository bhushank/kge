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
        self._entities = self._get_entities(triples)

    def _get_entities(self,data):
        entities  = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities)

    def sample(self,ex,num_samples,is_target=True):
        def get_candidates():
            return copy.copy(self._entities)

        samples = set()
        candidates = get_candidates()
        gold = ex.t if is_target else ex.s
        if gold in candidates:
            candidates.remove(gold)
        if len(candidates) <= num_samples:
            return candidates

        while True:
            idx = np.random.randint(0, len(candidates))
            p = Path(ex.s, ex.r, candidates[idx]) if is_target else  Path(candidates[idx], ex.r, ex.t)
            if p not in self._triples:
                    samples.add(candidates[idx])
            candidates.remove(candidates[idx])

            if len(samples) >= num_samples or len(candidates)<=0:
                return list(samples)


@util.memoize
def load_params(params_path,model):
    print("Loading Params from {}".format(params_path))

    with open(params_path,'r') as f:
        params = pickle.load(f)
    print("Finished Loading Params.")
    return Initialized(SparseParams(params),model.init_f)


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

