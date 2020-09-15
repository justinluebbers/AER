import sys
from util import parse_arguments
import data_manager as dm
import approximator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

###
# Conveniently train multiple configurations of Approximators
###
def main(env):
    train(
        model_name='EM_'+env, data_filename=env+'_transitions_1000000.p', 
        hwidths=[32,64,128,256,512,768], hdepths=[1,2,3,6], loss_function='nrmse_range', n_steps=50000, suc_ratio=0.0, dropout=0.1)
    # train(
    #     model_name='RM_'+env, data_filename=env+'_transitions_1000000.p', 
    #     hwidths=[2,4,8], hdepths=[0,2,4,8], loss_function='nrmse_range', n_steps=15000, suc_ratio=0.0, dropout=0.0)
    # train(
    #     model_name='TM_'+env, data_filename=env+'_transitions_1000000.p', 
    #     hwidths=[2,4,8], hdepths=[0,2,4,8], loss_function='termination_loss', n_steps=15000, suc_ratio=0.3, dropout=0.1)

def train(model_name, data_filename, hwidths, hdepths, loss_function, n_steps=20000, suc_ratio=0.0, dropout=0.1,load_model=None):
    n_steps = int(n_steps)
    suc_ratio = float(suc_ratio)
    hwidths = [int(w) for w in hwidths]
    hdepths = [int(d) for d in hdepths]
    dropout = float(dropout)

    input_keys = []
    output_keys = []

    if model_name.startswith('EM'):
        input_keys = ['o','u']
        output_keys = ['o_2']
    elif model_name.startswith('RM'):
        input_keys = ['o']
        output_keys = ['r']
    elif model_name.startswith('TM'):
        input_keys = ['o']
        output_keys = ['done']
    
    params = []
    for d in hdepths:
        for w in hwidths:
            params.append({'n_steps':n_steps, 'suc_ratio':suc_ratio, 'hwidth':w, 'hdepth':d,
                    'input_keys':input_keys, 'output_keys':output_keys,    
                    'learning_rate':0.001, 'dropout':dropout, 'loss_function': loss_function,
                    #'load_model':'app_MC_n20k_s0.1_h50x50_d0.0_lr0.0001_l-nrmse_range.ckpt',
                    'save_model':True, 'model_name':model_name, 'plot':False, 'save_losses':True,
                    'load_model':load_model
                    })

    run_approximator(params, data_filename)

def run_approximator(test_params, data_filename, only_success=False,**kwargs):
    DM = dm.DataManager('./AER/data/'+data_filename, dm.load_data, only_success=only_success,**test_params[0])
    for params in test_params:
        approx = approximator.Approximator(activation = tf.nn.leaky_relu, data_manager = DM, **params)
        approx.train(sample_function = approx.data_manager.sample_transitions_from_dict, **params)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main('MountainCar')
    else:
        args, kwargs = parse_arguments(sys.argv)
        train(**kwargs)

