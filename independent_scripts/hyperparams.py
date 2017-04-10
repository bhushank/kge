import numpy as np
import json


def main():
    data = 'freebase'
    base = "/home/mitarb/kotnis/Data/bordes/"
    bilinear('transE',data,base)

def bilinear(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}_{}".format(data,model) + "{}.json"
    config = json.load(open(path + exp_name.format("")))
    l2 = np.sort(np.random.uniform(1,5,size=5))
    count = 1
    for e in l2:
            config['l2'] = np.power(10,-e)
            json.dump(config,open(path+exp_name.format("_"+str(count)),'w'),
                      sort_keys=True,separators=(',\n', ':'))
            count+=1

def typeregularizer(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}_{}".format(data,model) + "{}.json"
    config = json.load(open(path + exp_name.format("")))

    l2 = np.sort(np.random.uniform(1,5,size=5))
    alpha = np.sort(np.random.uniform(1,5,size=5))
    count = 1
    for e in l2:
        for a in alpha:
            config['l2'] = np.power(10,-e)
            config['alpha'] = np.power(10,-a)
            json.dump(config,open(path+exp_name.format("_"+str(count)),'w'))
            count+=1

if __name__=='__main__':
    main()