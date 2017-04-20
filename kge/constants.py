__author__ = 'Bhushan Kotnis'

import util
'''Model Names'''

bilinear = 'bilinear'
s_rescal = 's-rescal'
hole = 'hole'
transE = 'transE'


'''Negatives'''
num_train_negs = 10
num_dev_negs = 100
num_test_negs = 1000#float('inf')

'''Dev Samples'''
num_dev_samples = 200

'''Parameter Scaling'''
param_scale = util.to_floatX(0.1)

'''SGD Batch Size'''
batch_size = 300
test_batch_size = 1000

'''Report and Save model'''
report_steps = 100
save_steps = 200

'''Early Stopping'''
early_stop_counter = 5

'''Position of positive'''
pos_position = 0

category='has_cat'
cat_file = 'entity_cat.cpkl'
#cat_file='entity_cat.cpkl'
