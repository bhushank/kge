__author__ = 'Bhushan Kotnis'
import numpy as np
from collections import MutableMapping
import copy


class SparseParams(MutableMapping):
    '''
    Contains dict, rather than inheriting from a dict. This allows for multiple views of same underlying dict.
    Also allows on to pickle the dict which can then be analysed later.
    '''

    def __init__(self,d=None):
        if d is None:
            d = {}
        assert isinstance(d,dict)
        # Should only be accessible through interface methods
        self.__d = d


    def __getitem__(self, key):
        return self.__d.__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(value,np.ndarray) or isinstance(value,float)
        self.__d.__setitem__(key,value)

    def __delitem__(self, key):
        return self.__d.__delitem__(key)

    def __contains__(self, key):
       return self.__d.__contains__(key)

    def __iter__(self):
        # Need to implement, else python will use __getitem__(n), n= 0,1,2...
        return iter(self.__d)

    def __len__(self):
        return self.__d.__len__()

    def as_dict(self):
        return self.__d

    def __str__(self):
        return str(self.__d)

    def __repr__(self):
        return repr(self.__d)


    def select(self, keys):
        '''
        Returns new sparse params for given keys
        :return:SparseParams
        '''
        sp = SparseParams()
        for key in keys:
            val = self.get(key)
            if val is not None:
                sp[key] = val

        return sp


    def remove(self,key):
        # Returning the object (self) allows for chaining
        return self.pop(key,None)

    def iapply(self,f):
        '''
        Apply function to parameters in place
        :param f:
        :return:
        '''

        for key, val in self.iteritems():
            self[key] = f(key,val)

        return self

    def _copy_op(self,op):
        new = copy.deepcopy(self)
        op(new)
        return new

    def apply(self,f):
        def op(x):
            self.iapply(f)
        return self._copy_op(op)

    def norm2(self):
        sq_sum = 0.0
        for v in self.itervalues():
            sq_sum += np.square(np.linalg.norm(v))
        return np.sqrt(sq_sum)

    def _isqr(self):
        for key in self.iterkeys():
            val = self[key]
            val = val*val
            self[key] = val

    def sqr(self):
        def op(x):
            x._isqr()
        return self._copy_op(op)

    def _isqrt(self):
        for key in self.iterkeys():
            val = self[key]
            val = np.sqrt(val)
            self[key] = val

    def sqrt(self):
        def op(x):
            x._isqrt()
        return self._copy_op(op)

    def __iadd__(self, other):

        for key,other_val in other.iteritems():
            val = self.get(key,None)
            if val is None:
                val = other_val
            else:
                val += other_val

            self[key] = val

        return self

    def __add__(self, other):

        def op(x):
            x += other

        return self._copy_op(op)

    def __imul__(self, other):
        isfloat = isinstance(other, float)
        remove_list = []
        for key,val in self.iteritems():
            other_val = other if isfloat else other.get(key,None)
            if other_val is None:
                remove_list.append(key)
            else:
                val*=other_val
                self[key] = val
        for e in remove_list:
            self.pop(e)
        return self

    def __mul__(self, other):
        def op(x):
            x*= other
        return self._copy_op(op)

    def __eq__(self, other):

        if not isinstance(other,SparseParams):
            return False
        # check keys
        if self.keys() != other.keys():
            return False

        for key in self.iterkeys():
            v1 = self[key]
            v2 = other[key]
            if isinstance(v1,np.ndarray) or isinstance(v2,np.ndarray):
                if any(v1 != v2):
                    return False
            else:
                if v1 != v2:
                    return False

        return True

    def __ne__(self, other):
        return not self == other

    def size(self):
        return len(self.__d)

    def __sizeof__(self):
        return len(self.__d)

    @classmethod
    def random(self,x,sigma=1.0):
        '''
        Generate Random Parameters with same shape as x
        :return:
        '''
        rand = SparseParams()
        for key, val in x.iteritems():
            if isinstance(val, float):
                d = np.random.normal(scale=sigma)
            else:
                d = np.random.normal(scale=sigma, size=x[key].shape)
            rand[key] = d
        return rand

class Initialized(SparseParams):
    '''
    Create a wrapper class that initializes parameters. This is not directly added to __getitem__
    because we want to throw exception during test time if no entity/relation found
    '''
    def __init__(self,sparse_params,init_f):
        SparseParams.__init__(self,sparse_params.as_dict())
        self.init_f = init_f

    def __getitem__(self, key):
        try:
            return SparseParams.__getitem__(self,key)
        except KeyError:
            val = self.init_f(key)
            if val is None:
                raise KeyError()
            self[key] = val
            return val