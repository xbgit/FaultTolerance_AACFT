import numpy as np
import random
from ftcode.utils.timer import timer
from ftcode.utils import binary_heap
import sys
import math

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done, fault_info):
        data = (obs_t, action, reward, obs_tp1, done, fault_info)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], {}
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.concatenate(obs_t[:]))
            actions.append(action)
            obses_tp1.append(np.concatenate(obs_tp1[:]))
            rewards.append(reward[:])
            dones.append(done[0])
            for k, v in info.items():
                if k not in infos.keys():
                    infos[k] = []
                infos[k].append(v)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), infos

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBufferPrioritized(object):
    def __init__(self, size, batch_size, n_agents):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = {}
        self._maxsize = int(size)
        self.n_agents = n_agents
        self.priority_queue = [binary_heap.BinaryHeap(self._maxsize) for i in range(n_agents + 1)]
        self._next_idx = 0
        self.alpha = 0.4
        self.epsilon = 1e-5
        self.partition_num = 100
        self.batch_size = batch_size
        self.distributions = self.build_distributions()

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self._next_idx % self._maxsize == 0:
            self._next_idx = 1
        else:
            self._next_idx += 1
        return self._next_idx

    def add(self, obs_t, action, reward, obs_tp1, done, fault_info):
        data = (obs_t, action, reward, obs_tp1, done, fault_info)
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._storage:
                del self._storage[insert_index]
            self._storage[insert_index] = data
            # add to priority queue
            for pq_index in range(len(self.priority_queue)):
                if pq_index < self.n_agents and fault_info['fault_list'][pq_index]:
                    continue

                priority = self.priority_queue[pq_index].get_max_priority()
                self.priority_queue[pq_index].update(priority, insert_index)
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        for pq_index in range(4):
            self.priority_queue[pq_index].balance_tree()

    def retrieve(self, sampled_indices):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], {}
        for i in sampled_indices:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.concatenate(obs_t[:]))
            actions.append(action)
            obses_tp1.append(np.concatenate(obs_tp1[:]))
            rewards.append(reward[:])
            dones.append(done[0])
            for k, v in info.items():
                if k not in infos.keys():
                    infos[k] = []
                infos[k].append(v)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), infos

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self._maxsize / n_partitions))

        for n in range(partition_size, self._maxsize + 1, partition_size):
            distribution = {}
            # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
            pdf = list(
                map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
            )
            pdf_sum = math.fsum(pdf)
            distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
            # split to k segment, and than uniform sample in each k
            # set k = batch_size, each segment has total probability is 1 / batch_size
            # strata_ends keep each segment start pos and end pos
            cdf = np.cumsum(distribution['pdf'])
            strata_ends = {1: 0, self.batch_size + 1: n}
            step = 1 / float(self.batch_size)
            index = 1
            for s in range(2, self.batch_size + 1):
                while cdf[index] < step:
                    index += 1
                strata_ends[s] = index
                step += 1 / float(self.batch_size)

            distribution['strata_ends'] = strata_ends

            res[partition_num] = distribution

            partition_num += 1

        return res

    def sample(self, beta, pq_index):
        """
                sample a mini batch from experience replay
                :param beta:
                :return: experience, list, samples
                :return: w, list, weights
                :return: rank_e_id, list, samples id, used for update priority
                """

        dist_index = math.floor(self.priority_queue[pq_index].size / self._maxsize * self.partition_num)
        # issue 1 by @camigord
        partition_size = math.floor(self._maxsize / self.partition_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]

        rank_list = []
        # sample from k segments
        for n in range(1, self.batch_size + 1):
            index = random.randint(distribution['strata_ends'][n] + 1, distribution['strata_ends'][n + 1])
            rank_list.append(index)

        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id

        rank_e_id = self.priority_queue[pq_index].priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return experience, w, rank_e_id

    def update_priority(self, indices, delta, pq_index):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue[pq_index].update(delta[i], indices[i])

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def collect(self):
        return self.sample(-1)


# class ReplayBufferPrioritized(object):
#     def __init__(self, size):
#         """Create Prioritized Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         self._storage = []
#         self.priority_queue = [binary_heap.BinaryHeap(int(size))] * 4
#         self._priorities = [[], [], [], []]
#         self._maxsize = int(size)
#         self._next_idx = 0
#         self.alpha = 0.4
#         self.epsilon = 1e-5
#
#     def __len__(self):
#         return len(self._storage)
#
#     def clear(self):
#         self._storage = []
#         self._next_idx = 0
#
#     @timer
#     def add(self, obs_t, action, reward, obs_tp1, done, fault_info):
#         data = (obs_t, action, reward, obs_tp1, done, fault_info)
#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#             for i in range(4):
#                 if i < 3 and fault_info['fault_list'][i]:
#                     priority = -1000
#                 else:
#                     priority = max(self._priorities[i]) if len(self._priorities[i]) > 0 else 1000
#                 self._priorities[i].append(priority)
#         else:
#             self._storage[self._next_idx] = data
#             for i in range(4):
#                 if i < 3 and fault_info['fault_list'][i]:
#                     priority = -1000
#                 else:
#                     priority = max(self._priorities[i])# if len(self._priorities[i]) > 0 else 1000
#                 self._priorities[i][self._next_idx] = priority
#         self._next_idx = (self._next_idx + 1) % self._maxsize
#
#     @timer
#     def sample(self, batch_size, beta, index):
#         priorities = np.array(self._priorities[index]).reshape(-1)  # add a small value epsilon to ensure numeric stability
#         if index < 3:
#             fault_list = (np.array(self._priorities[index]) == -1000)
#             min_pri = np.min(priorities[~fault_list])
#             if min_pri < 0:
#                 priorities += (-min_pri + self.epsilon)
#             priorities[fault_list] = 0
#             priorities = np.power(priorities, self.alpha)
#         else:
#             priorities = np.power(priorities + self.epsilon, self.alpha)
#         p = priorities/np.sum(priorities)  # compute a probability density over the priorities
#         sampled_indices = np.random.choice(np.arange(len(p)), size=batch_size, p=p)  # choose random indices given p
#         p = np.array([p[i] for i in sampled_indices]).reshape(-1)
#         weights = np.power(batch_size * p, -beta)
#         weights /= weights.max()
#
#         obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], {}
#         for i in sampled_indices:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done, info = data
#             obses_t.append(np.concatenate(obs_t[:]))
#             actions.append(action)
#             obses_tp1.append(np.concatenate(obs_tp1[:]))
#             rewards.append(reward[:])
#             dones.append(done[0])
#             for k, v in info.items():
#                 if k not in infos.keys():
#                     infos[k] = []
#                 infos[k].append(v)
#
#         return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), infos, weights, sampled_indices
#
#     def update(self, indices, priorities, index):
#         """Update the priority values after training given the samples drawn."""
#         for i, priority in zip(indices, priorities):
#             self._priorities[index][i] = priority
#
#     def make_index(self, batch_size):
#         return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#
#     def make_latest_index(self, batch_size):
#         idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
#         np.random.shuffle(idx)
#         return idx
#
#     def collect(self):
#         return self.sample(-1)
