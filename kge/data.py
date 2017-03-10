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

    def __init__(self,triples,typed=True,cats=None):
        self._triples = set(triples)
        self.typed = typed
        if typed:
            self._typed_negs = self._get_negs(triples)
        else:
            self._entities = self._get_entities(triples)
        if cats is not None:
            self._entity_cats = cats
            self._cat_entities = self._invert_dict(cats)
            self._test_cats = self._get_test_cats()


    def _get_negs(self,data):
        negatives = defaultdict(lambda : tuple([set(),set()]))
        for ex in data:
            r = ex.r[0]
            val = negatives[r]
            val[0].add(ex.s)
            val[1].add(ex.t)
            negatives[r] = val

        return negatives

    def _get_entities(self,data):
        entities  = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities)

    def sample(self,ex,num_samples,is_target=True):
        def get_candidates(r):
            '''
            Returns a copy of the candidates, so that sampled negatives
            can be removed from the copy without affecting state
            :param is_target: boolean
            :param r: string
            :return: candidates: set
            '''
            if self.typed:
                candidates = self._typed_negs[r]
                if is_target:
                    return copy.copy(candidates[1])
                else:
                    return copy.copy(candidates[0])
            return copy.copy(self._entities)

        samples = set()
        candidates = get_candidates(ex.r[0])
        if is_target:
            if ex.t in candidates:
                candidates.remove(ex.t)
        else:
            if ex.s in candidates:
                candidates.remove(ex.s)
        candidates = list(candidates)
        # source or target need not be present in candidates (dev data does not overlap with train)
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


    def get_typed_entities(self,categories):
        entities = set()
        for c in categories:
            entities.update(self._cat_entities.get(c,set()))
        return entities

    '''
    Negative Samples for Categories
    '''
    def _get_test_cats(self):
        '''
        Categories for test time, cannot use labelled data. Returns relation (range) category data
        :return:
        '''
        test_cats = dict()
        for ex in self._triples:
            cats = test_cats.get(ex.r[0], set())
            if ex.t in self._entity_cats:
                cats.update(self._entity_cats[ex.t])
            test_cats[ex.r[0]] = cats
        return test_cats

    def _invert_dict(self, d):
        inv_d = dict()
        for key, val in d.iteritems():
            for v in val:
                keys = inv_d.get(v, set())
                keys.add(key)
                inv_d[v] = keys
        return inv_d


    def sample_pos_cats(self,ex,is_target):
        if is_target:
            pos = self._entity_cats.get(ex.t,set())
        else:
            pos = self._entity_cats.get(ex.s,set())
        test_cats = self._test_cats[ex.r[0]]
        samples = test_cats.union(pos)
        return samples

    def sample_neg_cats(self, ex,is_target):
        '''
        Sample random categories and then add the NN categories as negatives. Remove all positives
        :param ex:
        :param is_target:
        :return:
        '''
        all_cats = set(self._cat_entities.keys())
        e = ex.t if is_target else ex.s
        pos_cats = self._entity_cats.get(e,set())
        candidates = list(all_cats.difference(set(pos_cats)))
        samples = set(
            [candidates[x] for x in np.random.choice(range(len(candidates)),
                                                     size=constants.num_dev_negs, replace=False)])
        test_cats = self._test_cats[ex.r[0]]
        samples.update(test_cats.difference(pos_cats))
        intersect = samples.intersection(pos_cats)
        assert len(intersect) == 0
        return list(samples)


@util.memoize
def load_params(params_path,model):
    print("Loading Params from {}".format(params_path))

    with open(params_path,'r') as f:
        params = pickle.load(f)
    print("Finished Loading Params.")
    return Initialized(SparseParams(params),model.init_f)


def read_dataset(path,dev_mode=True,max_examples = float('inf'),is_path_data=False,is_cat=False):

    data_set = {}
    train = 'train_no_cats' if is_cat else 'train'
    data_set['train'] = read_file(os.path.join(path,train),max_examples,is_path_data)
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

@util.memoize
def load_categories(cat_path):
    print("Loading categories from {}".format(cat_path))

    with open(cat_path) as f:
        cats = pickle.load(f)
    print("Finished loading category data")
    return cats

