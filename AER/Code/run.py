import sys
import data_manager as dm
import approximator
from util import parse_arguments
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

###
# Convenient way of starting Approximator training
###

# Train an approximator alias environment model
def run_approximator(test_params, data_filename, only_success=False,**kwargs):
    DM = dm.DataManager('./AER/data/'+data_filename, dm.load_data, only_success=only_success,**test_params[0])
    for params in test_params:
        approx = approximator.Approximator(activation = tf.nn.leaky_relu, data_manager = DM, **params)
        approx.train(sample_function = approx.data_manager.sample_transitions_from_dict, **params)

def main(args):
    args, kwargs = parse_arguments(args)
    run_approximator(**kwargs)

if __name__ == "__main__":
    main(sys.argv)