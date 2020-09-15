from run import run_approximator
import sys
from util import parse_arguments

import os
m = os.getcwd()

###
# Conveniently train multiple configurations of Approximators
###

def main(model_name, data_filename, hwidths, hdepths, loss_function, n_steps=20000, suc_ratio=0.0, dropout=0.1,load_model=None):
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

if __name__ == '__main__':
    args, kwargs = parse_arguments(sys.argv)
    main(**kwargs)


