__author__ = 'Bhushan Kotnis'

import util
import cPickle as pickle
import os
import numpy as np
from collections import defaultdict
from parameters import SparseParams,Initialized
import copy

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

    def __init__(self,triples,typed=True):
        self._triples = set(triples)
        self.typed = typed
        if typed:
            self._typed_negs = self._get_negs(triples)
        else:
            self._entities = self._get_entities(triples)

    def _get_entities(self,data):
        entities  = set()
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

    def _get_candidates(self,is_source,r):
        '''
        Returns a copy of the candidates, so that sampled negatives
        can be removed from the copy without affecting state
        :param is_source: boolean
        :param r: string
        :return: candidates: set
        '''
        if self.typed:
            candidates = self._typed_negs[r]
            if is_source:
                return copy.copy(candidates[0])
            else:
                return copy.copy(candidates[1])
        return copy.copy(self._entities)

    def sample(self,ex,is_source,num_samples):
        samples = set()
        candidates = list(self._get_candidates(is_source,ex.r[0]))
        if len(candidates) <= 1:
            return samples
        if len(candidates) <= num_samples:
            return candidates

        while True:
            idx = np.random.randint(0, len(candidates))
            p = Path(candidates[idx], ex.r, ex.t) if is_source else Path(ex.s, ex.r, candidates[idx])
            if p not in self._triples:
                    samples.add(candidates[idx])
                    candidates.remove(candidates[idx])

            if len(samples) <= num_samples:
                return list(samples)





@util.memoize
def load_params(params_path,objective):
    print("Loading Params from {}".format(params_path))

    with open(params_path,'r') as f:
        params = pickle.load(f)
    print("Finished Loading Params.")
    return Initialized(SparseParams(params),objective.init_f)


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



