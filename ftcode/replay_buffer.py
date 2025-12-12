import numpy as np
from ftcode.utils.timer import timer


class ReplayBuffer:
    """Base class for storing and sampling transition data in multi-agent environments.

    Stores observations, actions, rewards, next observations, done flags, and additional info.
    Supports batch addition of transitions and random batch sampling.
    """

    def __init__(self, size, obs_shapes, action_shapes):
        """Initialize the replay buffer.

        Args:
            size: Maximum capacity of the buffer
            obs_shapes: List of observation dimensions for each agent, e.g., [[dim1], [dim2], ...]
            action_shapes: List of action dimensions for each agent
        """
        self.max_size = int(size)
        self.obs_shapes = obs_shapes
        self.n_agents = len(obs_shapes)

        # Calculate index ranges for each agent's observations and actions in concatenated arrays
        self.obs_indices = self._compute_indices(obs_shapes)
        self.action_indices = self._compute_indices(action_shapes)

        # Pre-allocate memory for core transition data
        total_obs_dim = sum(obs_shapes)
        total_action_dim = sum(action_shapes)
        self.obs_t = np.zeros((self.max_size, total_obs_dim), dtype=np.float32)  # Observations at time t
        self.actions = np.zeros((self.max_size, total_action_dim), dtype=np.float32)  # Actions taken
        self.rewards = np.zeros((self.max_size, self.n_agents), dtype=np.float32)  # Rewards received
        self.obs_tp1 = np.zeros_like(self.obs_t)  # Observations at time t+1
        self.dones = np.zeros(self.max_size, dtype=np.bool_)  # Termination flags

        # Storage for additional information (e.g., fault status)
        self.infos = None

        # Buffer management variables
        self.next_idx = 0
        self.size = 0

    def _compute_indices(self, shapes):
        """Compute index ranges for each agent in concatenated arrays.

        Args:
            shapes: List of dimensions (one per agent)

        Returns:
            List of (start, end) index tuples for each agent
        """
        indices = []
        start = 0
        for s in shapes:
            end = start + s
            indices.append((start, end))
            start = end
        return indices

    def add(self, obs_t_batch, action_batch, reward_batch, obs_tp1_batch, done_batch, fault_info_batch=None):
        """Add a batch of transitions to the buffer.

        Args:
            obs_t_batch: Observations at time t, shape (n_rollout, n_agent, obs_dim)
            action_batch: Actions taken, shape (n_rollout, n_agent, action_dim)
            reward_batch: Rewards received, shape (n_rollout, n_agent)
            obs_tp1_batch: Observations at time t+1, shape (n_rollout, n_agent, obs_dim)
            done_batch: Termination flags, shape (n_rollout,)
            fault_info_batch: Optional list of dictionaries containing additional info,
                              each with shape (n_rollout, ...)
        """
        batch_size = obs_t_batch.shape[0]
        # Calculate indices with wrap-around if buffer is full
        indices = np.arange(self.next_idx, self.next_idx + batch_size) % self.max_size

        # Copy observations and actions using vectorized operations
        self._batch_copy(self.obs_t, indices, obs_t_batch, self.obs_indices)
        self._batch_copy(self.actions, indices, action_batch, self.action_indices)
        self._batch_copy(self.obs_tp1, indices, obs_tp1_batch, self.obs_indices)

        # Store rewards and termination flags
        self.rewards[indices] = reward_batch
        self.dones[indices] = done_batch[:, 0]

        # Handle additional information
        if self.infos is None:
            self._init_infos(fault_info_batch)
        # Convert list of dicts to dict of arrays for storage
        converted_info = {}
        for k in self.infos.keys():
            converted_info[k] = np.array([info[k] for info in fault_info_batch])
        # Store converted info
        for k in converted_info:
            self.infos[k][indices] = converted_info[k]

        # Update buffer state
        self.next_idx = (self.next_idx + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def _batch_copy(self, target, indices, src_batch, indices_list):
        """Helper method to batch copy agent-specific data to concatenated arrays.

        Args:
            target: Target array to store copied data
            indices: Indices in target array to write to
            src_batch: Source batch data, shape (batch_size, n_agents, ...)
            indices_list: List of (start, end) indices for each agent
        """
        for a in range(self.n_agents):
            start, end = indices_list[a]
            target[indices, start:end] = src_batch[:, a, :]

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple containing:
                - obs_t: Observations at time t
                - actions: Actions taken
                - rewards: Rewards received
                - obs_tp1: Observations at time t+1
                - dones: Termination flags
                - infos: Additional information dictionary
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Zero-copy batch extraction
        batch = (
            self.obs_t[indices],
            self.actions[indices],
            self.rewards[indices],
            self.obs_tp1[indices],
            self.dones[indices],
            {k: v[indices] for k, v in self.infos.items()}
        )
        return batch

    def _init_infos(self, sample_info):
        """Initialize storage for additional information based on first batch.

        Args:
            sample_info: First batch of information to determine shapes and types
        """
        first_info = sample_info[0]
        self.infos = {}

        for k in first_info.keys():
            # Determine data type and shape from first sample
            sample_value = first_info[k]
            dtype = type(sample_value) if not isinstance(sample_value, (list, np.ndarray)) else np.asarray(
                sample_value).dtype
            # Create storage array with shape (max_size, ...)
            shape = (self.max_size,) + np.asarray(sample_value).shape
            self.infos[k] = np.zeros(shape, dtype=dtype)
