import numpy as np
import random

###
import pickle
###

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
    ###
    # , save_path):
    ###
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        ###
        # self.save_buffer = False
        # if save_path is not None:
        #     self.save_buffer = True
        #     self.save_path = save_path
        #     self.transition_storage = []
        ###

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        ###
        # if self.save_buffer and self._next_idx == 0:
        #     self.transition_storage.extend([
        #         {'o': obs[0], 'u': np.array([acts]), 'r': rew, 'o_2': next_obs[0], 'done': np.array([done])}
        #         for (obs, acts, rew, next_obs, done) in self._storage
        #     ])
        ###
    
    ###
    # def save_transitions(self):
    #     if self._next_idx != 0:
    #         self.transition_storage.extend([
    #             {'o': obs[0], 'u': np.array([acts]), 'r': rew, 'o_2': next_obs[0], 'done': np.array([done])}
    #             for (obs, acts, rew, next_obs, done) in self._storage[0:self._next_idx]
    #         ])
    #     pickle.dump(self.transition_storage, open(self.save_path, 'wb'))
    ###

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    ###
    # def _encode_sample_aer(self, idxes, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, _, reward, _, _ = data
    #         #obs_t: (1,2), action: int, done: float
    #         d = np.random.randint(aer_d)
    #         obs_tp1 = obs_t
    #         for _ in range(d+1):
    #             obs_t = obs_tp1
    #             inp = list(obs_t[0])
    #             action = act(np.array(obs_t[0])[None])[0]
    #             inp.append(action)
    #             done = AER_sess.run(TM, feed_dict={TM_Z:[inp]})[0]
    #             done = round(done[0])*1.0
    #             obs_tp1 = AER_sess.run(EM, feed_dict={EM_Z:[inp]})
    #         #obs_tp1: (1,2), done:(1,)
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    # def sample_aer(self, aer_batch_size, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z):
    #     idxes = [random.randint(0, len(self._storage) - 1) for _ in range(aer_batch_size)]
    #     return self._encode_sample_aer(idxes, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z)

    # def _encode_sample_aer_gan(self, f_obs, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z, GM, GM_Z):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for o in f_obs:
    #         #obs_t: (1,2), action: int, done: float
    #         d = np.random.randint(aer_d)
    #         o = [o]
    #         obs_tp1 = o
    #         for _ in range(d+1):
    #             obs_t = obs_tp1
    #             inp = list(obs_t[0])
    #             action = act(np.array(obs_t[0])[None])[0]
    #             inp.append(action)
    #             done = AER_sess.run(TM, feed_dict={TM_Z:[inp]})[0]
    #             done = round(done[0])*1.0
    #             obs_tp1 = AER_sess.run(EM, feed_dict={EM_Z:[inp]})
    #         #obs_tp1: (1,2), done:(1,)
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append([-1.0])
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    
    # def sample_aer_gan(self, aer_batch_size, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z, GM, GM_Z):
    #     f_obs = AER_sess.run(GM, feed_dict={GM_Z: np.random.uniform(-1., 1., size=[aer_batch_size, GM_Z.shape.dims[1].value])})
    #     return self._encode_sample_aer_gan(f_obs, aer_d, act, AER_sess, EM, EM_Z, TM, TM_Z, GM, GM_Z)
###

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
