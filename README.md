# Knowledge Graph Embedding with Type Regularizer


This code implements the state-of-the-art Knowledge Graph Embedding [algorithms](http://www.cs.technion.ac.il/~gabr/publications/papers/Nickel2016RRM.pdf) such as [RESCAL](http://www.dbs.ifi.lmu.de/~tresp/papers/p271.pdf), [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data), [HOLE](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828), and S-RESCAL with added Type Regularizer for Freebase. Computational Graphs are implemented using [Theano](http://deeplearning.net/software/theano/). 

# Installation
## Data
* Create a data directory and download FB15K datasets from [here](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz)
* Rename freebase_mtr100_mte100-train.txt, freebase_mtr100_mte100-valid.txt, freebase_mtr100_mte100-test.txt as train, dev, test respectively. These files should reside inside the data directory.


