import numpy as np
from ftcode.fault.registry import FAULTS
from ftcode.fault.nofault import NoFault
import random
import copy

@FAULTS.register
class Broken(NoFault):
    def __init__(self, env, alg_controller, cl_controller):
        super().__init__(env, alg_controller, cl_controller)
        self.cl_controller = cl_controller
        self.alg_controller = alg_controller

        # self.fault_probs = [0, 1/2, 1/2]
        self.fault_probs = [1/3, 1/3, 1/3]
        self.fault_id = -1
        self.episode_cnt = 0
        self.fault_time = -1

    def add_fault(self, time_step, obs_n):
        # if no agent is failed
        if time_step == 0:
            self.fault_time = self.cl_controller.get_fault_time(self.episode_cnt)
        if not any(self.fault_list):
            if time_step >= self.fault_time:
                self.fault_id = random.choices(self.agent_list, self.fault_probs)[0]
                self.fault_list[self.fault_id] = True
                self.agents[self.fault_id].fault = True
                self.fault_change = True
            else:
                self.fault_change = False
        else:
            self.fault_change = False


    def obs_fault(self, obs_n):
        if self.fault_id >= 0:
            self.alg_controller.obs_fault_modify(obs_n, self.fault_list)

    def action_fault(self, action_n):
        if self.fault_id >= 0:
            action_n[self.fault_id] = np.zeros(len(action_n[self.fault_id]))

    def new_obs_fault(self, new_obs_n):
        if self.fault_id >= 0:
            self.alg_controller.obs_fault_modify(new_obs_n, self.fault_list)

    def reset(self):
        super().reset()
        self.fault_id = -1
        self.episode_cnt += 1

    def info(self):
        return {'fault_list': copy.deepcopy(self.fault_list), 'fault_change': copy.deepcopy(self.fault_change)}


