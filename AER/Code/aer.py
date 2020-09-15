from gan import GAN
from approximator import Approximator

import functools
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

###
# Implementation of the AER algorithm
###

class AER():
    def __init__(self, kwargs, full_rollouts=False):
        # self.init_params(kwargs, env)
        aer_params = kwargs['use_aer']
        del kwargs['use_aer']
        self.em_file = aer_params[0]
        self.rm_file = aer_params[1]
        self.tm_file = aer_params[2]
        self.rollout_depth = aer_params[3]
        self.batch_size = aer_params[4]
        self.log_mean_reward_freq = aer_params[5]

        self.full_rollouts = full_rollouts

    # Initialize
    def init_params(self, sess, env):
        self.sess = sess
        self.env = env

        
        em_depth, em_width = self.em_file.split('h')[1].split('_')[0].split('x')
        rm_depth, rm_width = self.rm_file.split('h')[1].split('_')[0].split('x')
        tm_depth, tm_width = self.tm_file.split('h')[1].split('_')[0].split('x')
        em = Approximator(hdepth = int(em_depth), hwidth = int(em_width))
        rm = Approximator(hdepth = int(rm_depth), hwidth = int(rm_width))
        tm = Approximator(hdepth = int(tm_depth), hwidth = int(tm_width))
        self.obs_size = functools.reduce(lambda x,y:x*y, self.env.observation_space.shape) if self.env.observation_space.shape is not () else 1
        self.acts_size = functools.reduce(lambda x,y:x*y, self.env.action_space.shape) if self.env.action_space.shape is not () else 1
        self.EM, self.EM_Z = em.load_session(self.sess, filename=self.em_file, input_size=self.obs_size+self.acts_size, output_size=self.obs_size)
        self.RM, self.RM_Z = rm.load_session(self.sess, filename=self.rm_file, input_size=self.obs_size, output_size=1)
        self.TM, self.TM_Z = tm.load_session(self.sess, filename=self.tm_file, input_size=self.obs_size, output_size=1)
        # self.TM = tf.round(tf.nn.sigmoid(self.TM))


        # self.gan = False
        # if 'use_gan' in kwargs:
        #     gan_params = kwargs['use_gan']
        #     del kwargs['use_gan']

        #     self.gan = True

        #     gm_file = gan_params[0]
        #     gm_depth, gm_width = gm_file.split('g')[1].split('-')[0].split('x')
        #     gm = GAN(g_hdepth = int(gm_depth), g_hwidth = int(gm_width))

        #     self.GM, self.GM_Z = gm.load_session(self.sess, filename=gm_file, output_size=self.obs_size)

    # Ensure access to the agent's policy to choose actions in artificial scenarios
    def init_policy(self, policy):
        self.policy = policy

    # Ensure access to the experience buffer for selecting starting states
    def init_experience_access(self, experience_access):
        self.get_observation_batch = experience_access

    # Sample data from the models
    def sample_transitions(self):
        if self.full_rollouts:
            return self._sample_aer_full_rollouts()
        else:
            return self._sample_aer()

    # Deprecated
    def _sample_aer(self):
        observations = self.get_observation_batch(self.batch_size)
        #idxes = [random.randint(0, len(observations) - 1) for _ in range(self.batch_size)]
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for obs_t in observations:
            #data = storage[i]
            #obs_t, _, reward, _, _ = data
            #obs_t: (1,2), action: int, done: float
            d = np.random.randint(self.rollout_depth)
            obs_tp1 = obs_t.flatten()
            for _ in range(d+1):
                obs_t = obs_tp1.flatten()
                inp = list(obs_t)
                action, info = self.policy(obs_t)
                inp.append(action)
                
                obs_tp1 = self.sess.run(self.EM, feed_dict={self.EM_Z:[inp]})[0]
                done = self.sess.run(self.TM, feed_dict={self.TM_Z:[inp]})[0]
                done = round(done[0])*1.0
                # RM einfügen
                reward = np.array([-1])
            #obs_tp1: (1,2), done:(1,)
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(infos)

    # Sample complete trajectories of artificial data
    def _sample_aer_full_rollouts(self):
        observations = self.get_observation_batch(self.batch_size)
        #idxes = [random.randint(0, len(observations) - 1) for _ in range(self.batch_size)]
        obs_rollouts, action_rollouts, reward_rollouts, next_obs_rollouts, done_rollouts, info_rollouts = [], [], [], [], [], []
        #for obs_t in observations:
            #data = storage[i]
            #obs_t, _, reward, _, _ = data
            #obs_t: (1,2), action: int, done: float
        rollout_obs = observations.tolist()
        rollout_actions = [[] for _ in range(self.batch_size)]
        rollout_dones = [[] for _ in range(self.batch_size)]
        rollout_rewards = [[] for _ in range(self.batch_size)]
        rollout_infos = [[] for _ in range(self.batch_size)]

        # obs_tp1 = obs_t.flatten()
        for _ in range(self.rollout_depth):
            # obs_t = obs_tp1.flatten()
            actions,infos,current_obs = [],[],[]

            for i in range(self.batch_size):
                current_obs.append(rollout_obs[i][-1])
                action, info = self.policy(rollout_obs[i][-1])
                actions.append(action)
                infos.append(info)
            
            inputs = []
            for i in range(self.batch_size):
                inputs.append(current_obs[i].copy())
                inputs[-1].append(actions[i])
            
            next_obs, dones, rewards = self.sess.run([self.EM, self.TM,self.RM], feed_dict={self.EM_Z:inputs,self.TM_Z:current_obs,self.RM_Z:current_obs})
            # next_obs = self.sess.run(self.EM, feed_dict={self.EM_Z:inputs})
            # dones = self.sess.run(self.TM, feed_dict={self.TM_Z:current_obs})
            # rewards = self.sess.run(self.RM, feed_dict={self.RM_Z:current_obs})

            ### auch langsam
            # next_2obs, don2es, re2wards = self.sess.run([tf.zeros([4,2],tf.float32),tf.zeros([4,1],tf.float32),tf.zeros([4,1],tf.float32)])
            # next_obs = self.sess.run(tf.zeros([16,2],tf.float32))
            # dones = self.sess.run(tf.zeros([16,1],tf.float32))
            # rewards = self.sess.run(tf.zeros([16,1],tf.float32))

            # next_obs = np.zeros(shape=[4,2])
            # dones = np.zeros(shape=[4,1])
            # rewards = np.zeros(shape=[4,1])

            for i in range(self.batch_size):
                
                rollout_obs[i].append(next_obs[i].tolist())
                rollout_actions[i].append(actions[i])
                rollout_dones[i].append(dones[i].tolist())
                rollout_rewards[i].append(rewards[i].tolist())
                rollout_infos[i].append(infos[i])

        #obs_tp1: (1,2), done:(1,)
        obs_rollouts = np.array(rollout_obs)[:,:-1,:].reshape([self.batch_size*self.rollout_depth,1,self.obs_size])
        next_obs_rollouts = np.array(rollout_obs)[:,1:,:].reshape([self.batch_size*self.rollout_depth,1,self.obs_size])
        action_rollouts = np.array(rollout_actions).reshape([self.batch_size*self.rollout_depth])
        reward_rollouts = np.array(rollout_rewards).reshape([self.batch_size*self.rollout_depth,1])
        done_rollouts = np.array(rollout_dones).reshape([self.batch_size*self.rollout_depth])
        info_rollouts = np.array(rollout_infos).reshape([self.batch_size*self.rollout_depth])
        return obs_rollouts, action_rollouts, reward_rollouts, next_obs_rollouts, done_rollouts, info_rollouts
