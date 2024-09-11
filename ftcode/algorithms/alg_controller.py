from ftcode.replay_buffer import ReplayBuffer
import numpy as np
from ftcode.utils.timer import timer

class AlgController:
    def __init__(self, args, env_args, ex_name):
        self.args = args
        self.memory = ReplayBuffer(args.memory_size)
        self.obs_shape_n = env_args['obs_shape_n']
        self.action_shape_n = env_args['action_shape_n']
        self.obs_size, self.action_size = AlgController.shape2size(self.obs_shape_n, self.action_shape_n)
        self.n_agents = len(self.obs_shape_n)

    @staticmethod
    def shape2size(obs_shape_n, action_shape_n):
        obs_size = []
        action_size = []
        head_o, head_a, end_o, end_a = 0, 0, 0, 0
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            end_o = end_o + obs_shape
            end_a = end_a + action_shape
            range_o = (head_o, end_o)
            range_a = (head_a, end_a)
            obs_size.append(range_o)
            action_size.append(range_a)
            head_o = end_o
            head_a = end_a
        return obs_size, action_size

    def update(self, args, episode_id):
        pass
        # self.update_transition()

    def alg_info2metrics(self, episode_info):
        pass

    def update_transition(self, obs_n, new_obs_n, rew_n, done_n, action_n, fault_info_n):
        if not fault_info_n['fault_change']:
            self.memory.add(obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n, fault_info_n)

    def policy(self, observations, training_mode=True):
        action_probs = self.joint_action_probs(self.current_histories, training_mode)
        return [np.random.choice(self.actions, p=probs) for probs in action_probs]
        # self.current_histories = self.extend_histories(self.current_histories, observations)
        # action_probs = self.joint_action_probs(self.current_histories, training_mode)
        # return [numpy.random.choice(self.actions, p=probs) for probs in action_probs]

    # def save_model(self):
    #     pass
    #
    # def load_model(self):
    #     pass
