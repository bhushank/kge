__author__ = 'Bhushan Kotnis'
import os
import cPickle as pickle
def main():
    base = '/home/mitarb/kotnis/'
    triples = base+ '/Data/bordes/freebase/train_cats'
    entities = read_train_entities(triples)
    ent_cats = read_categories(base,entities)
    add_to_train(triples,ent_cats)

def read_train_entities(file_path):
    entities = set()
    with open(file_path) as f:
        for line in f:
            s,r,t = line.strip().split('\t')
            entities.add(s)
            entities.add(t)
    return entities

def read_categories(base,entities):
    entity_cat = dict()
    path = base + '/pra/data/relation_metadata/freebase/category_instances/'
    for f in os.listdir(path):
        print("Processing file {}".format(f))
        with open(path+f) as data:
            for line in data:
                e = line.rstrip()
                if e in entities:
                    cats = entity_cat.get(e,set())
                    cats.add(f)
                    entity_cat[e] = cats
    print("Number of entities with category data {}".format(len(entity_cat.keys())))
    print('Writing to pickle...')
    pickle.dump(entity_cat,open('entity_cat.cpkl','w'))
    print('Done')
    return entity_cat

def add_to_train(triples,entity_cat):
    with open(triples,'a') as f:
        for e,cats in entity_cat.iteritems():
            for c in cats:
                f.write("{}\thas_category\t{}\n".format(e,c))


if __name__=='__main__':
    main()