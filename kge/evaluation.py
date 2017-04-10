import numpy as np
import constants
from data import Path
import util
import time
from sys import stdout
import os
class Evaluater(object):
    def __init__(self,model,neg_sampler):
        self.neg_sampler = neg_sampler
        self.model = model

    def evaluate(self,batch,num_negs):
        raise NotImplementedError()

class RankEvaluater(Evaluater):
    def __init__(self,model,neg_sampler):
        super(RankEvaluater,self).__init__(model,neg_sampler)
        self.init_score = float('inf') # because lower the better
        self.metric_name = "Mean Rank"
        self.tol = 0.1

    def comparator(self,curr_score,prev_score):
        # write if curr_score less than prev_score
        return curr_score < prev_score + self.tol

    def evaluate(self,batch,num_negs):

        pos_scores = self.model.predict(batch).data.cpu().numpy()
        mean_rank = []
        for p,ex in zip(pos_scores,batch):
            negs = self.neg_sampler.sample(ex,num_negs,True)
            neg_batch = [Path(ex.s,ex.r,n) for n in negs]
            scores = self.model.predict(neg_batch).data.cpu().numpy().flatten()
            scores = np.append(scores,p)
            ranks = util.ranks(scores, ascending=False)
            mean_rank.append(ranks[-1])


        return np.nanmean(mean_rank)


class TestEvaluater(Evaluater):
    def __init__(self,model,neg_sampler,params,is_dev,results_dir):
        super(TestEvaluater,self).__init__(model,neg_sampler)
        self.cache = {}
        self.all_ranks = []
        self.is_dev = is_dev
        self.results_dir =results_dir
        self.params = params

    def write_ranks(self):
        all_ranks = [str(x) for x in self.all_ranks]
        with open(os.path.join(self.results_dir, 'ranks_checkpoint'), 'w') as f:
            f.write("\n".join(all_ranks))

    def metrics(self,rank,rr,hits_10,ind):
        rr = (rr * ind + 1.0 / float(rank)) / float(ind + 1)
        h_10 = 1.0 if rank <= 10 else 0.0
        hits_10 = (hits_10 * ind + h_10) / float(ind + 1)
        self.all_ranks.append(rank)
        return rr,hits_10

    def pack_negs(self,ex,negs,is_target):
        batch = []
        for n in negs:
            p = Path(ex.s,ex.r,n) if is_target else Path(n,ex.r,ex.t)
            batch.append(p)

        return batch

    def evaluate(self,batch,num_negs):
        #print("Prediciting Positives")
        rep_steps = 10
        #avg_quantile = 0.0
        rr = 0.0
        hits_10 = 0
        pos = self.model.predict(self.params,batch).ravel()
        ind = 0
        #print("Predicting Negatives")
        start = time.time()
        for ex, p in zip(batch, pos):
            s_rank,t_rank = self.compute_metrics(p,ex)
            rr,hits_10 = self.metrics(s_rank,rr,hits_10,ind)
            ind += 1
            rr, hits_10 = self.metrics(t_rank, rr, hits_10, ind)
            ind += 1
            if ind % (rep_steps*2) == 0:
                end = time.time()
                secs = (end - start)
                stdout.write("\rSpeed {} qps. Percentage complete {}, MRR {}, HITS@10 {} ".
                             format(rep_steps / float(secs), 0.5*ind/float(constants.test_batch_size) * 100,rr,hits_10))
                stdout.flush()
                start = time.time()
        self.write_ranks()
        return rr, hits_10



    def compute_metrics(self,pos,ex):


        def calc_scores(batches):
            scores = []
            for b in batches:
                scores.append(self.model.predict(self.params,b))
            scores = np.array(scores).ravel()
            scores = np.append(scores,pos)
            assert pos == scores[-1]
            ranks = util.ranks(scores.flatten(), ascending=False)
            return ranks[-1]

        if self.is_dev:
            s_negs = self.neg_sampler.sample(ex,1000, False)
            t_negs = self.neg_sampler.sample(ex,1000, True)
        else:
            s_negs= self.neg_sampler.bordes_negs(ex,False,1000)
            t_negs = self.neg_sampler.bordes_negs(ex,True,1000)
        #num_s_negs = len(s_negs)
        #num_t_negs = len(t_negs)

        s_negs  = self.pack_negs(ex, s_negs,False)
        t_negs  = self.pack_negs(ex, t_negs, True)

        negs_s = util.chunk(s_negs,constants.test_batch_size)
        negs_t = util.chunk(t_negs, constants.test_batch_size)
        s_rank = calc_scores(negs_s)
        t_rank = calc_scores(negs_t)
        #print("{},{}".format(s_rank,t_rank))
        return s_rank,t_rank