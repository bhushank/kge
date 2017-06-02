# Knowledge Graph Embeddings with Type Regularizer


This code implements the state-of-the-art Knowledge Graph Embedding [algorithms](http://www.cs.technion.ac.il/~gabr/publications/papers/Nickel2016RRM.pdf) such as [RESCAL](http://www.dbs.ifi.lmu.de/~tresp/papers/p271.pdf), [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data), [HOLE](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828), and S-RESCAL with added Type Regularizer for Freebase. Computational Graphs are implemented using [Theano](http://deeplearning.net/software/theano/). 

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
"is_dev": True if you want to test on validation data (must be specified),
"is_typed": True if you want to use Type Regularizer (default False),
"alpha": Strength of Type Regularizer (default 1.0),
"l2": Strength of L2 regularizer (must be specified),
"model": Model (bilinear, transE, s-rescal, hole) (must be specified),
"num_epochs": Max number of epochs (default 100),
"param_scale": Parameter Initialization Scale (default 0.1)


