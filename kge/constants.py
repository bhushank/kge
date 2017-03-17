__author__ = 'Bhushan Kotnis'

import util
'''Model Names'''

bilinear = 'bilinear'
s_rescal = 's-rescal'
hole = 'hole'
transE = 'transE'
tr = 'tr'

'''Negatives'''
num_dev_negs = 10
num_test_negs = 200

'''Dev Samples'''
num_dev_samples = 200

'''Parameter Scaling'''
param_scale = util.to_floatX(0.1)

'''SGD Batch Size'''
batch_size = 300
test_batch_size = 100

'''Report and Save model'''
report_steps = 10
save_steps = 20

'''Early Stopping'''
early_stop_counter = 5

'''Position of positive'''
pos_position = 0

