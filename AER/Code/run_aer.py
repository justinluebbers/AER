import os

import baselines.run as run
import pickle
from util import parse_arguments
import sys
from env_training_parameters import get_model_names

###
# Convenient way to run experiments with different configurations
###

import multiprocessing as mp

DIR = './AER/'

# Construct parameter list for baselines.run for AER
def make_params(env, alg, em_name, rm_name, tm_name, 
                log_path, save_path, batch_size, depth, 
                num_timesteps = 10000, gan = False, log_freq=1000,
                priotitized_replay=False,
                **kwargs):
    param = []
    param.append('./baselines/baselines/run.py')
    param.append('--alg={}'.format(alg))
    param.append('--env={}'.format(env))
    param.append('--num_timesteps={}'.format(num_timesteps))
    param.append("--use_aer=['{}', '{}', '{}', {}, {}, {}]".format(em_name, rm_name, tm_name, depth, batch_size, log_freq))
    param.append("--log_path={}".format(log_path))
    param.append("--save_path={}/model".format(save_path))
    param.append("--prioritized_replay={}".format(priotitized_replay))
    for key in kwargs:
        param.append("--{}={}".format(key,kwargs[key]))
    return param

# Construct parameter list for baselines.run for DQN benchmarks
def make_default_params(env, alg, log_path, save_path, num_timesteps = 10000, batch_size = None, priotitized_replay=False, **kwargs):
    param = []
    param.append('./baselines/baselines/run.py')
    param.append('--alg={}'.format(alg))
    param.append('--env={}'.format(env))
    param.append('--num_timesteps={}'.format(num_timesteps))
    if batch_size is not None:
        param.append('--batch_size={}'.format(batch_size))
    param.append("--use_aer=['{}', '{}', '{}', {}, {}, {}]".format('', '', '', 0, 0, 1000))
    param.append("--log_path={}".format(log_path))
    param.append("--save_path={}/model".format(save_path))
    param.append("--prioritized_replay={}".format(priotitized_replay))
    for key in kwargs:
        param.append("--{}={}".format(key,kwargs[key]))
    return param

# Task for one worker for parallel execution of experiments
def worker(param, q=None):
    _,rewards = run.main(param)
    q.put(rewards)

# Manage file system AER
def make_path(DIR, env, alg, num_timesteps, batch_size, depth, prioritized_replay=False, perfect_models=[]):
    path1 = DIR
    
    perfect_str = 'perfect'
    for s in perfect_models:
        perfect_str += '_{}'.format(s)
    perfect_str += '/'

    path2 = '/{}/{}{}/{}/{}bs{}/d{}/'.format(env, alg, 
        '/per' if prioritized_replay else '/default_replay', 
        num_timesteps, 'non_perfect/' if perfect_models == [] else perfect_str, batch_size, depth)
    def foo(folder):
        return path1 + folder + path2
    return foo

# Manage file system DQN benchmarks
def make_default_path(DIR, env, alg, num_timesteps, batch_size, prioritized_replay=False):
    path1 = DIR
    path2 = '/{}/{}{}/{}/default/bs{}/'.format(env, alg, '/per' if prioritized_replay else '/default_replay', num_timesteps, batch_size)
    def foo(folder):
        return path1 + folder + path2
    return foo

# Run experiments
def main(env, rollouts=[], iterations=5, num_timesteps=None, real_batch_sizes=[], alg='deepq', gan=False, prioritized_replay=False, perfect_models=[], **kwargs):
    iterations = int(iterations)
    env_name = env.split('-')[0]

    if num_timesteps is not None:
        num_timesteps = int(num_timesteps)
        em_name, rm_name, tm_name, _ = get_model_names(env_name, perfect_models)
    else:
        em_name, rm_name, tm_name, num_timesteps = get_model_names(env_name, perfect_models)
    
    for batch_size in real_batch_sizes:
        if batch_size == '': continue
        batch_size = int(batch_size)
        p = make_default_path(DIR, env_name, alg, num_timesteps, batch_size, prioritized_replay=prioritized_replay)
        log_path = p('logs')
        save_path = p('models')
        reward_path = p('scores')

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(reward_path):
            os.makedirs(reward_path)

        q = mp.Queue()
        
        jobs = []
        for i in range(iterations):
            param = make_default_params(env=env, alg=alg,
                log_path="{}{}/".format(log_path,i), save_path="{}{}/".format(save_path,i), 
                num_timesteps=num_timesteps, batch_size=batch_size, priotitized_replay=prioritized_replay,
                **kwargs)
            
            p = mp.Process(target=worker, args=(param, q))
            # p = mp.Process(target=worker, args=([param]))
            jobs.append(p)
            p.start()
        
        for job in jobs:
            job.join()

        reward_records = []
        for i in range(iterations):
            reward_records.append(q.get())
        pickle.dump(reward_records, open(reward_path+'rewards_i{}_n{}_bs{}{}.p'.format(iterations, num_timesteps, batch_size, '_per' if prioritized_replay else ''), 'wb'))
        
        
        


    for sd in rollouts:
        if sd == '': continue
        batch_size, depth = sd.split('x')   
        batch_size = int(batch_size)
        depth = int(depth)

        p = make_path(DIR, env_name, alg, num_timesteps, batch_size, depth, prioritized_replay=prioritized_replay, perfect_models=perfect_models)
        log_path = p('logs')
        save_path = p('models')
        reward_path = p('scores')

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(reward_path):
            os.makedirs(reward_path)

        q = mp.Queue()
        jobs = []
        for i in range(iterations):
            param = make_params(env=env, alg=alg, em_name=em_name, rm_name=rm_name, tm_name=tm_name,
                    log_path="{}{}/".format(log_path,i), save_path="{}{}/".format(save_path,i), 
                    batch_size=batch_size, depth=depth, num_timesteps=num_timesteps, gan=gan, 
                    priotitized_replay=prioritized_replay, **kwargs)
                
            p = mp.Process(target=worker, args=(param, q))
            # p = mp.Process(target=worker, args=([param]))
            jobs.append(p)
            p.start()
        
        for job in jobs:
            job.join()
    
        reward_records = []
        for i in range(iterations):
            reward_records.append(q.get())
        pickle.dump(reward_records, open(reward_path+'rewards_i{}_n{}_bs{}_d{}{}.p'.format(iterations, num_timesteps, batch_size, depth, '_per' if prioritized_replay else ''), 'wb'))

if __name__ == '__main__':
    args, kwargs = parse_arguments(sys.argv)
    main(**kwargs)

