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

'''Report and Save model'''
report_steps = 20
save_steps = 10

'''Early Stopping'''
early_stop_counter = 5

'''Position of positive'''
pos_position = 0


'''Universal Category'''
universal_cat = 'object'
cat_rel = 'has_category'