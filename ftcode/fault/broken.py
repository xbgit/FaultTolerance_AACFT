import numpy as np
from ftcode.fault.registry import FAULTS
from ftcode.fault.nofault import NoFault
import random
import copy


@FAULTS.register
class Broken(NoFault):
    """Fault controller for scenarios with agent faults (e.g., sudden failure).

    Extends the NoFault controller to introduce configurable agent faults during episodes.
    Faulty agents are marked as non-movable and non-collidable, with their actions nullified.
    """

    def __init__(self, args, env, cl_controller):
        """Initialize the broken fault controller.

        Args:
            args: Configuration parameters
            env: Environment instance
            cl_controller: Curriculum learning controller for fault timing
        """
        super().__init__(args, env, cl_controller)
        self.cl_controller = cl_controller  # For determining fault timing

        # Fault probability configuration
        self.fault_probs = args.fault_probs
        if self.fault_probs == 'equal':
            self.fault_probs = np.ones(self.n) / self.n  # Equal probability for all agents

        self.fault_id = -1  # Current faulty agent index (-1 = no fault)
        self.episode_cnt = 0  # Number of episodes completed
        self.fault_time_start = args.fault_time  # Base fault start time
        self.fault_time = -1  # Scheduled fault times for current episode
        self.time_step = -1  # Current environment time step
        self.fault_max = args.fault_max  # Maximum number of faults per episode

    def add_fault(self, time_step):
        """Add faults to agents based on scheduled timing and fault limits.

        Updates fault statuses when the current time step reaches scheduled fault times.
        Marks faulty agents as non-movable and non-collidable.

        Args:
            time_step: Current environment time step
        """
        self.time_step = time_step

        # Initialize fault times for new episodes
        if time_step == 0:
            self.fault_time = [self.cl_controller.get_fault_time(self.episode_cnt)
                               for _ in range(self.fault_max)]
            self.fault_time.sort()  # Ensure fault times are in order

        # Add new faults if under max fault limit and at scheduled time
        if np.sum(self.fault_list) < self.fault_max:
            current_fault_idx = np.sum(self.fault_list)
            if time_step >= self.fault_time[current_fault_idx]:
                # Determine which agent to fault
                self.fault_id = random.choices(self.agent_list, self.fault_probs)[0]

                # Update fault status
                self.fault_list[self.fault_id] = True
                self.agents[self.fault_id].fault = True
                self.agents[self.fault_id].movable = False
                self.agents[self.fault_id].collide = False
                self.fault_change = True
            else:
                self.fault_change = False
        else:
            self.fault_change = False

    def reset(self):
        """Reset fault tracking for a new episode.

        Resets all agents to non-faulty state and increments episode counter.
        """
        super().reset()
        self.fault_time = -1
        self.fault_id = -1
        self.episode_cnt += 1

    def info(self):
        """Return detailed fault information for the current step.

        Returns:
            Dictionary with fault status list, change flag, scheduled times, and current time
        """
        return {
            'fault_list': copy.deepcopy(self.fault_list),
            'fault_change': copy.deepcopy(self.fault_change),
            'fault_time': self.fault_time,
            'current_time': self.time_step
        }

    @staticmethod
    def obs_fault(obs_n, fault_info, func_obs_fault_modify):
        """Modify observations to reflect agent faults.

        Applies observation modifications using a provided function to mask/replace
        observations related to faulty agents.

        Args:
            obs_n: List of observations for each agent
            fault_info: Fault status information
            func_obs_fault_modify: Function to apply observation modifications
        """
        for env_idx, info in enumerate(fault_info):
            func_obs_fault_modify(obs_n[env_idx], info['fault_list'])

    def action_fault(self, action_n):
        """Nullify actions of faulty agents.

        Sets actions of currently identified faulty agents to zero.

        Args:
            action_n: List of actions from each agent (modified in-place)
        """
        if self.fault_id >= 0:
            action_n[self.fault_id][:] = 0

    @staticmethod
    def new_obs_fault(new_obs_n, fault_info, func_obs_fault_modify):
        """Modify new observations (post-action) to reflect agent faults.

        Similar to obs_fault but applied to observations after an action is taken.

        Args:
            new_obs_n: List of new observations after action
            fault_info: Fault status information
            func_obs_fault_modify: Function to apply observation modifications
        """
        for env_idx, info in enumerate(fault_info):
            func_obs_fault_modify(new_obs_n[env_idx], info['fault_list'])

    @staticmethod
    def action_fault_static(action_n, fault_info):
        """Static method to nullify actions of faulty agents.

        Sets actions of currently identified faulty agents to zero (supports batched environments).

        Args:
            action_n: List of actions from each agent (modified in-place)
            fault_info: List of fault status dictionaries (one per environment)
        """
        for env_idx, info in enumerate(fault_info):
            for agent_idx, action in enumerate(action_n):
                if info['fault_list'][agent_idx]:
                    action[env_idx][:] = 0