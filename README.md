# Knowledge Graph Embeddings with Type Regularizer


This code implements the state-of-the-art Knowledge Graph Embedding [algorithms](http://www.cs.technion.ac.il/~gabr/publications/papers/Nickel2016RRM.pdf) such as [RESCAL](http://www.dbs.ifi.lmu.de/~tresp/papers/p271.pdf), [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data), [HOLE](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828), and S-RESCAL with added Type Regularizer for Freebase. Computational Graphs are implemented using [Theano](http://deeplearning.net/software/theano/).   Available SGD Algorithms : ADAM, Adagrad. Algorithms are hand coded and implementing SGD variations should be straightforward.

# Installation
## Data
* Create a data directory and download FB15K datasets from [here](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz)
* Rename freebase_mtr100_mte100-train.txt, freebase_mtr100_mte100-valid.txt, freebase_mtr100_mte100-test.txt as train, dev, test respectively. These files should reside inside the data directory.

## Configuration
* Create directory ./data/experiment_specs/
* Experiment configurations are specified as JSON files inside the experiment_specs directory.
* An example configuration file (freebase_bilinear.json) can be found in the repository.
### Specifications
* "batch_size": Train Batch Size (default 300),
* "entity_dim": Embedding Dimension (must be specified),
* "exp_name": Experiment Name,
* "is_dev": True if you want to test on validation data (must be specified),
* "is_typed": True if you want to use Type Regularizer (default False),
* "alpha": Strength of Type Regularizer (default 1.0),
* "l2": Strength of L2 regularizer (must be specified),
* "model": Model (bilinear, transE, s-rescal, hole) (must be specified),
* "num_epochs": Max number of epochs (default 100),
* "param_scale": Parameter Initialization Scale (default 0.1)  
Additional specifications (early stopping, save and report time, etc) can be changed by modifying constants.py

# Usage
To train and test simply run  
*python experiment_runner.py "data_dir" "experiment_name"*  
where experiment_name is the name of the JSON file without the .json extension. For example  
*python experiment_runner.py ./data/ freebase_bilinear*

# Extending this
* Implementing new model
  * Implement Theano models in theano_models.py
  * Implement bprop, cost (fprop) and init_f for parameter initialization functions in model.py. Additionally if your model has a different architecture then you may need to implement gradient collection and data unpacking methods.
* Implementing SGD Algorithm
  * Extend the Updater class in Algorithms.py and add an update method.


