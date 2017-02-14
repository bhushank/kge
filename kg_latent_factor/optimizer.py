
from parameters import SparseParams,Initialized
import numpy as np
import util
import cPickle as pickle
import os
import time

class GradientDescent(object):
    def __init__(self,train,dev,updater,objective,evaluater,results_dir,model_type,config,init_params=None):

        self.train = train
        self.dev = dev
        self.batch_size = config['batch_size']
        self.l2_reg = config['l2_reg']

        if not init_params:
            init_params = SparseParams(d=dict())

        #param which is not available will be initialized
        self.params = Initialized(init_params,objective.init_f)

        self.results_dir = results_dir
        self.model_type = model_type # Single or Coupled
        # updater, cost, evaluator functions
        self.objective = objective
        self.updater = updater # updates at each iteration
        self.evaluater = evaluater # evaluates F-Score or Mean Average Precision or Rank
        # For reporting and saving params
        self.report_steps = 100#config['report_steps']
        self.save_steps = 1000#config['save_steps']
        self.dev_samples = config['num_dev_samples'] # number of dev samples
        # History
        self.history = {}
        self.grad_norm_history = []
        self.grad_history = []
        # For profiling
        self.prev_steps = 0
        self.prev_time = time.time()
        #Early stopping
        self.halt = False
        self.prev_score = evaluater.init_score
        self.early_stop_counter = config['early_stop']
        self.early_stop = config['early_stop']
        self.max_steps = config['max_steps']

    def l2reg_grad(self,grad):
        l2_grad = SparseParams(dict())
        for key in grad:
            if key in self.params:
                rgrad = self.params[key]*-2.0*self.l2_reg
                l2_grad[key] = rgrad

        return l2_grad


    def unit_norm(self,grad):

        for feature in grad:
            # Only for entities
            if feature[0]=='e':
                self.params[feature] *= ( 1.0 / np.linalg.norm(self.params[feature]) )

    def grad_clipper(self,grad):

        norm = grad.norm2()
        self.grad_norm_history.append(norm)
        # take median of maximum of last 1000 steps
        size = len(self.grad_norm_history)
        self.grad_norm_history = self.grad_norm_history[size - min(size,1000) :]
        self.grad_norm_history.sort()
        # Use median, because the number of parameters in each example could be different.
        # Median requires sorting, but python uses timsort which works faster on almost-sorted arrays.
        # We use median because it is robust to outliers.
        median = self.grad_norm_history[int(len(self.grad_norm_history)/2)]
        thresh = 3.0 * median
        if norm > thresh:
            grad*= median / norm

        return grad

    def minimize(self):
        self.steps = 0
        #rand = np.random.RandomState(2568)
        self.save()
        while True:
            train_cp = list(self.train)
            np.random.shuffle(train_cp)
            batches = util.chunk(train_cp, self.batch_size)

            for batch in batches:
                grad = SparseParams(d=dict())
                for p in batch:
                    grad_p = self.objective.gradient(self.params,p)
                    if grad_p.size() > 0:
                        grad += grad_p

                if self.l2_reg!=0:
                    grad += self.l2reg_grad(grad)

                # grad normalization by batch size
                grad *= 1.0 / len(batch)

                self.gnorm = grad.norm2()
                # updater algorithm, ADAM, RMSProp, etc
                delta = self.updater.update(grad,self.steps)
                # Gradient clipping
                delta = self.grad_clipper(delta)
                # Update
                self.params += delta
                self.steps += 1
                # Make sure all entity params have unit norms. unit_norm changes self.params
                self.unit_norm(delta)
                #Reports progress
                self.report(delta)
                #Writes params to disk periodically and determines stopping criterion
                self.save()

                if self.halt:
                    return

    def calc_obj(self,data, f,sample=True,is_rank = False, num_samples=-1):
        if sample:
            samples = util.sample(data, self.dev_samples)
        else:
            samples = data
        if is_rank:
            values = [f(self.params, x,num_samples) for x in samples]
        else:
            values = [f(self.params,x) for x in samples]
        return np.nanmean(values)





    def save(self):
        if self.steps % self.save_steps == 0:
            curr_score = self.calc_obj(self.dev,self.evaluater.evaluate,False,True,200)
            epochs = float(self.steps * self.batch_size) / len(self.train)
            print 'steps: {}, epochs: {:.2f}'.format(self.steps, epochs)
            print("Current Score: {}, Previous Score: {}".format(curr_score,self.prev_score))
            if self.evaluater.comparator(curr_score, self.prev_score):
                print("Saving params...")

                # Write history of objective func to disk
                with open(os.path.join(self.results_dir,'history_{}.cpkl'.format(self.model_type)),'w') as f:
                    pickle.dump(self.history,f)

                # Write parameters to disk
                with open(os.path.join(self.results_dir,'params_{}.cpkl'.format(self.model_type)),'w') as f:
                    pickle.dump(self.params.as_dict(),f)
                self.prev_score = curr_score
                # Reset early stop counter
                self.early_stop_counter = self.early_stop
            else:
                self.early_stop_counter -= 1
                print("New params worse than current, skip saving...")
            # Stopping Criterion, do at least 4 epochs
            if epochs > 4.0:
                if self.early_stop_counter == 0 or self.steps > self.max_steps:
                    self.halt = True


    def report(self,delta):

        if self.steps % self.report_steps == 0:
            grad_norm = self.gnorm
            delta_norm = delta.norm2()
            norm_rep = "Gradient Norm: {:.3f}, Delta Norm: {:.3f}".format(grad_norm,delta_norm)
            # Profiler
            secs = time.time() - self.prev_time
            num_steps = self.steps - self.prev_steps
            speed = num_steps / float(secs)
            self.prev_steps = self.steps
            self.prev_time = time.time()
            speed_rep = "Speed: {:.2f} steps/sec".format(speed)
            # Objective
            train_obj = self.calc_obj(self.train,self.objective.cost)
            dev_obj = self.calc_obj(self.train,self.objective.cost)
            obj_rep = "Train Obj: {:.3f}, Dev Obj: {:.3f}".format(train_obj,dev_obj)
            # Add to history
            self.history[time.time()] = (train_obj,dev_obj)
            # Performance
            train_val = self.calc_obj(self.train, self.evaluater.evaluate)
            dev_val = self.calc_obj(self.dev, self.evaluater.evaluate)
            metric = self.evaluater.metric_name
            eval_rep = "Train {} {:.3f}, Dev {} {:.3f}".format(metric,train_val,metric, dev_val)
            print("{}, {}, {}, {}".format(norm_rep,speed_rep,obj_rep,eval_rep))