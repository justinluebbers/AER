from train_approximator import main as train
import sys
from util import parse_arguments

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

if __name__ == "__main__":
    args, kwargs = parse_arguments(sys.argv)
    main(**kwargs)
    