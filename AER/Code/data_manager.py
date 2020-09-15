import pickle
import numpy as np
import random

###
# Class for managing access to expert data for Approximator training
###

class DataManager:
    def __init__(self, filename, load_function, input_keys=['o','u'], output_keys=['o_2','r','done'],**kwargs):
        self.reload_data(filename,load_function, input_keys, output_keys)

    def reload_data(self, filename, load_function, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.data, self.data_suc, self.input_size, self.output_size = load_function(filename, input_keys, output_keys)
    
    ### sample transitions from data
    # batch_size: Number of transitions to sample
    # suc_ratio: Ratio of successful transition that should be in cluded in the training dataset
    def sample_transitions_from_dict(self, batch_size, suc_ratio=0.0):
        batch_suc = int(batch_size*suc_ratio)
        suc_indices = np.random.choice(self.data_suc['o'].shape[0], batch_suc)
        indices = np.random.choice(self.data['o'].shape[0], batch_size-batch_suc)
        input_data, labels = [],[]
        for idx in indices:
            new_input = []
            for key in self.input_keys:
                new_input.extend(self.data[key][idx])
            input_data.append(np.array(new_input))
            new_label = []
            for key in self.output_keys:
                new_label.extend(self.data[key][idx])
            labels.append(np.array(new_label))
        for idx in suc_indices:
            new_input = []
            for key in self.input_keys:
                new_input.extend(self.data_suc[key][idx])
            input_data.append(np.array(new_input))
            new_label = []
            for key in self.output_keys:
                new_label.extend(self.data_suc[key][idx])
            labels.append(np.array(new_label))
        return np.array(input_data), np.array(labels)


### Load the training data from given file
# filename: Name of the file that contains the data
# input_keys: List of keys that represent the input of the trained model
# output_keys: List of keys that represent the output of the trained model            
def load_data(filename, input_keys, output_keys, **kwargs):
    transitions = pickle.load(open(filename, 'rb'))
    transition_dict = {k:[] for k in transitions[0]}
    transition_dict_suc = {k:[] for k in transitions[0]}
    for transition in transitions:
        # hard coded, bound to done
        for key in transition_dict:
            transition_dict[key].append(transition[key])
        #if transition['o'][0] > MC_GOAL_POSITION and transition['o'][1] > MC_GOAL_VELOCITY:
        if transition['done'] == 1.0:
            for key in transition_dict_suc:
                transition_dict_suc[key].append(transition[key])
    for key in transition_dict:
        transition_dict[key] = np.array(transition_dict[key])
    for key in transition_dict_suc:
        transition_dict_suc[key] = np.array(transition_dict_suc[key])
    input_size = 0
    output_size = 0
    for k in input_keys:
        input_size += transitions[0][k].shape[0]
    for k in output_keys:
        output_size += transitions[0][k].shape[0]
    return transition_dict,transition_dict_suc,input_size,output_size

