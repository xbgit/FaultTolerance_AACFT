import torch
import numpy as np
from ftcode.replay_buffer import ReplayBuffer


class AlgController:
    """Base class for algorithm controllers.

    Provides core functionality for multi-agent reinforcement learning algorithms,
    including replay buffer management, observation/action shape handling, and
    basic training loop scaffolding.
    """

    def __init__(self, args, env_args, ex_name):
        """Initialize algorithm controller.

        Args:
            args: Configuration parameters
            env_args: Environment parameters (observation shapes, action shapes)
            ex_name: Experiment name for logging/saving
        """
        self.args = args
        self.obs_shape_n = env_args['obs_shape_n']  # Observation shape for each agent
        self.action_shape_n = env_args['action_shape_n']  # Action shape for each agent
        self.n_agents = len(self.obs_shape_n)  # Number of agents

        # Initialize replay buffer
        self.memory = ReplayBuffer(
            size=args.memory_size,
            obs_shapes=self.obs_shape_n,
            action_shapes=self.action_shape_n
        )

        # Compute cumulative size ranges for observations/actions (for concatenation)
        self.obs_size, self.action_size = self.shape2size(self.obs_shape_n, self.action_shape_n)

    @staticmethod
    def shape2size(obs_shape_n, action_shape_n):
        """Compute cumulative index ranges for observations and actions.

        Used to concatenate/extract multi-agent observations/actions.

        Args:
            obs_shape_n: List of observation shapes per agent
            action_shape_n: List of action shapes per agent

        Returns:
            Tuple of (obs_size, action_size), where each is a list of (start, end) tuples
            indicating the index range for each agent in concatenated arrays.
        """
        obs_size = []
        action_size = []
        obs_start, action_start = 0, 0

        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            obs_end = obs_start + obs_shape
            action_end = action_start + action_shape
            obs_size.append((obs_start, obs_end))
            action_size.append((action_start, action_end))
            obs_start, action_start = obs_end, action_end

        return obs_size, action_size

    def get_models(self):
        """Return all neural network models (to be implemented by subclasses)."""
        pass

    def prep_training(self, device):
        """Prepare models for training (e.g., set to train mode)."""
        pass

    def prep_rollouts(self, device):
        """Prepare models for rollouts (e.g., set to eval mode)."""
        pass

    def obs_fault_modify(self, obs, fault_list):
        """Modify observations based on agent faults (to be implemented by subclasses)."""
        pass

    def update(self, args, episode_id):
        """Perform one training iteration (to be implemented by subclasses)."""
        pass

    def alg_info2metrics(self, episode_info):
        """Log algorithm-specific metrics (to be implemented by subclasses)."""
        pass

    def update_transition(self, obs_old, obs_new, rewards, dones, actions, fault_info):
        """Add transition to replay buffer.

        Args:
            obs_old: Old observations (before action)
            obs_new: New observations (after action)
            rewards: Rewards received
            dones: Whether episode ended
            actions: Actions taken
            fault_info: Dictionary with fault status information
        """
        self.memory.add(obs_old, actions, rewards, obs_new, dones, fault_info)

    def update_transition_mp(self, batch_data):
        """Add multiple transitions to replay buffer (multi-processing).

        Args:
            batch_data: List of transition dictionaries
        """
        for data in batch_data:
            if not data['fault_info']['fault_change']:  # Skip if fault status changed
                self.memory.add(
                    data['obs'], data['action'], data['reward'],
                    data['new_obs'], data['done'], data['fault_info']
                )

    def policy(self, observations, fault_controller, fault_info, training_mode=True):
        """Generate actions for all agents (to be implemented by subclasses).

        Args:
            observations: List of observations per agent
            fault_controller: Fault management controller
            fault_info: Dictionary with fault status
            training_mode: Whether to use stochastic (train) or deterministic (eval) policy

        Returns:
            List of actions per agent
        """
        # Placeholder implementation
        action_probs = self.joint_action_probs(self.current_histories, training_mode)
        return [np.random.choice(self.actions, p=probs) for probs in action_probs]