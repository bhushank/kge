__author__ = 'Bhushan Kotnis'
from parameters import SparseParams,Initialized
import numpy as np
import util
import cPickle as pickle
import os
import time
import constants
import copy

class GradientDescent(object):
    def __init__(self,train,dev,updater,model,evaluater,results_dir,model_type,config,init_params=None,is_typed=False,typed_data=None):

        self.train = train
        self.dev = dev
        self.batch_size = config.get('batch_size',constants.batch_size)
        self.l2_reg = config['l2']
        self.alpha = float(config.get('alpha', 1.0))
        self.is_typed = is_typed
        self.typed_data = typed_data

        print("Setting L2 regularizer {:.5f}".format(self.l2_reg))
        if is_typed:
            print("Setting Type Regularizer {:.5f}".format(config.get('alpha',1.0)))
            print("Total Typed Triples {}".format(len(typed_data)))
        if not init_params:
            init_params = SparseParams(d=dict())

        #param which is not available will be initialized
        self.params = Initialized(init_params,model.init_f)

        self.results_dir = results_dir
        self.model_type = model_type # Single or Coupled
        # updater, cost, evaluator functions
        self.model = model
        self.updater = updater # updates at each iteration
        self.evaluater = evaluater # evaluates F-Score or Mean Average Precision or Rank
        # For reporting and saving params
        self.report_steps = constants.report_steps
        self.save_steps = constants.save_steps
        self.dev_samples = constants.num_dev_samples # number of dev samples
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
        self.early_stop_counter = constants.early_stop_counter
        self.patience = constants.early_stop_counter
        self.max_epochs = config['num_epochs']

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

    def sgd(self,batches,is_tr=False):
        alpha = self.alpha if is_tr else 1.0
        for batch in batches:
            grad = self.model.gradient(self.params, batch)
            if self.l2_reg != 0:
                grad += self.l2reg_grad(grad)

            # grad normalization by batch size (or multiply by TR coeff)
            grad *= alpha / len(batch)
            self.gnorm = grad.norm2()
            # updater algorithm, ADAM, RMSProp, etc
            delta = self.updater.update(grad, self.steps)
            # Gradient clipping
            delta = self.grad_clipper(delta)
            # Update
            self.params += delta
            self.steps += 1
            # Make sure all entity params have unit norms. unit_norm changes self.params
            self.unit_norm(delta)

            # Reports progress
            self.report(delta,is_tr)
            # Writes params to disk periodically and determines stopping criterion
            self.save()
            if self.halt:
                return

    def minimize(self):
        self.steps = 0
        #rand = np.random.RandomState(2568)
        self.save()
        train_cp = list(self.train)
        if self.is_typed and self.alpha==1.0:
            train_cp.extend(self.typed_data)
            print("alpha 1.0, combining training data, current training data triples {}".format(len(train_cp)))
            self.is_typed = False
        while True:
            if self.is_typed:
                typed_cp = list(self.typed_data)
                np.random.shuffle(typed_cp)
                typed_batches = util.chunk(typed_cp, self.batch_size)
                # For typed regularizer
                self.sgd(typed_batches, True)

            np.random.shuffle(train_cp)
            batches = util.chunk(train_cp, self.batch_size)
            self.sgd(batches)
            if self.halt:
                return

    def calc_obj(self,data, f,sample=True):
        if sample:
            samples = util.chunk(util.sample(data, self.dev_samples),100)
        else:
            samples = util.chunk(data,100)

        values = [f(self.params,np.asarray(s)) for s in samples]
        return np.nanmean(values)


    def save(self):

        if self.steps % self.save_steps == 0:
            self.evaluater.num_negs = constants.num_dev_negs
            curr_score = self.calc_obj(self.dev,self.evaluater.evaluate,True)
            data_size = len(self.train) + len(self.typed_data) if self.is_typed else len(self.train)
            epochs = float(self.steps * self.batch_size) / data_size
            print 'steps: {}, epochs: {:.2f}'.format(self.steps, epochs)
            print("Current Score: {}, Previous Score: {}".format(curr_score,self.prev_score))
            if self.evaluater.comparator(curr_score, self.prev_score) or epochs<1:
                print("Saving params...")
                # Write history of objective func to disk
                with open(os.path.join(self.results_dir,'history_{}.cpkl'.format(self.model_type)),'w') as f:
                   pickle.dump(self.history,f)

                # Write parameters to disk
                with open(os.path.join(self.results_dir,'params_{}.cpkl'.format(self.model_type)),'w') as f:
                    pickle.dump(self.params.as_dict(),f)
                self.prev_score = curr_score
                # Reset early stop counter
                self.early_stop_counter = copy.copy(self.patience)
            else:
                self.early_stop_counter -= 1
                print("New params worse than current, skip saving...")
            # Stopping Criterion, do at least 4 epochs
            if epochs >= 4.0:
                if self.early_stop_counter <= 0 or epochs >= self.max_epochs:
                    self.halt = True


    def report(self,delta,is_tr=False):

        if self.steps % (self.report_steps) == 0:
            self.evaluater.num_negs = constants.num_dev_negs
            grad_norm = self.gnorm
            delta_norm = delta.norm2()
            title = "Typed Reg." if is_tr else ""
            norm_rep = "{}Gradient Norm: {:.3f}, Delta Norm: {:.3f}".format(title,grad_norm,delta_norm)
            # Profiler
            secs = time.time() - self.prev_time
            num_steps = self.steps - self.prev_steps
            speed = num_steps / float(secs)
            self.prev_steps = self.steps
            self.prev_time = time.time()
            speed_rep = "Speed: {:.2f} steps/sec".format(speed)
            # Objective
            train_obj = self.calc_obj(self.train,self.model.cost)
            dev_obj = self.calc_obj(self.train,self.model.cost)
            obj_rep = "Train Obj: {:.3f}, Dev Obj: {:.3f}".format(train_obj,dev_obj)
            # Add to history
            #self.history[time.time()] = (train_obj,dev_obj)
            # Performance
            train_val = self.calc_obj(self.train, self.evaluater.evaluate)
            dev_val = self.calc_obj(self.dev, self.evaluater.evaluate)
            metric = self.evaluater.metric_name
            eval_rep = "Train {} {:.3f}, Dev {} {:.3f}".format(metric,train_val,metric, dev_val)
            print("{}, {}, {}, {}".format(norm_rep,speed_rep,obj_rep,eval_rep))



