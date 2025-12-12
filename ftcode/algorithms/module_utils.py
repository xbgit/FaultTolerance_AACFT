import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat


class CriticMLP(nn.Module):
    """MLP-based critic network for value function approximation.

    Inputs: Concatenated observations and actions of all agents
    Output: Estimated Q-value
    """

    def __init__(self, obs_shape, action_shape, hidden_size):
        super(CriticMLP, self).__init__()
        self.obs_shape = obs_shape  # Total observation dimension across all agents
        self.action_shape = action_shape  # Total action dimension across all agents
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Network layers
        self.fc1 = nn.Linear(action_shape + obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc_out.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """Forward pass to compute Q-value.

        Args:
            obs_input: Tensor of shape (batch_size, obs_shape)
            action_input: Tensor of shape (batch_size, action_shape)

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        # Concatenate observations and actions
        x = torch.cat([obs_input[:, 0:self.obs_shape], action_input[:, 0:self.action_shape]], dim=1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        return self.fc_out(x)


class ActorMLP(nn.Module):
    """MLP-based actor network for policy approximation.

    Input: Agent's observation
    Output: Stochastic policy (action probabilities)
    """

    def __init__(self, obs_dim, action_dim, hidden_size):
        super(ActorMLP, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.01)

        # Network layers
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, action_dim)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.fc_out.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, model_original_out=False):
        """Forward pass to compute policy.

        Args:
            obs_input: Tensor of shape (batch_size, obs_dim)
            model_original_out: If True, return raw logits + policy; else return policy

        Returns:
            If model_original_out: (logits, policy)
            Else: policy (action probabilities)
        """
        x = self.leaky_relu(self.fc1(obs_input))
        x = self.leaky_relu(self.fc2(x))
        logits = self.fc_out(x)

        # Gumbel-softmax for stochastic policy
        u = torch.rand_like(logits)
        policy = F.softmax(logits - torch.log(-torch.log(u)), dim=-1)

        if model_original_out:
            return logits, policy
        return policy


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.

    Computes attention weights as (QK^T / sqrt(d_k)) and applies to values.
    """

    def forward(self, query, key, value, mask=None):
        d_k = query.size()[-1]  # Dimension of query/key vectors
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled attention scores

        # Apply mask (e.g., for padding or causality)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights.matmul(value)  # Weighted sum of values


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Splits inputs into multiple heads, computes attention in parallel, then concatenates results.
    """

    def __init__(self, in_features, head_num, bias=False, activation=None):
        """
        Args:
            in_features: Input feature dimension (must be divisible by head_num)
            head_num: Number of attention heads
            bias: Whether to use bias in linear layers
            activation: Activation function applied after linear layers
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(f"`in_features`({in_features}) must be divisible by `head_num`({head_num})")

        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias

        # Linear layers for query, key, value, and output
        self.fc_q = nn.Linear(in_features, in_features, bias)
        self.fc_k = nn.Linear(in_features, in_features, bias)
        self.fc_v = nn.Linear(in_features, in_features, bias)
        self.fc_out = nn.Linear(in_features, in_features, bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.fc_q.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc_k.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc_v.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.1)

    def forward(self, q, k, v, mask=None):
        """Forward pass for multi-head attention.

        Args:
            q: Query tensor (batch_size, seq_len, in_features)
            k: Key tensor (batch_size, seq_len, in_features)
            v: Value tensor (batch_size, seq_len, in_features)
            mask: Attention mask (batch_size, seq_len, seq_len)

        Returns:
            Output tensor after attention (batch_size, seq_len, in_features)
        """
        # Linear projections + activation
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        # Reshape for multi-head processing
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        # Apply mask to all heads
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)

        # Compute scaled dot-product attention
        attn_output = ScaledDotProductAttention()(q, k, v, mask)

        # Reshape back to original dimensions
        attn_output = self._reshape_from_batches(attn_output)

        # Final linear projection
        output = self.fc_out(attn_output)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def scores(self, q, k, v):
        """Compute attention scores (for analysis/debugging)."""
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        d_k = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(d_k)
        return F.softmax(scores, dim=-1)

    @staticmethod
    def gen_history_mask(x):
        """Generate mask to only attend to past timesteps (causal mask)."""
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        """Reshape tensor to process each head in parallel.

        Input: (batch_size, seq_len, in_features)
        Output: (batch_size * head_num, seq_len, in_features / head_num)
        """
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        """Reshape tensor back to original dimensions after multi-head processing.

        Input: (batch_size * head_num, seq_len, in_features / head_num)
        Output: (batch_size, seq_len, in_features)
        """
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return f"in_features={self.in_features}, head_num={self.head_num}, bias={self.bias}, activation={self.activation}"


class Coder(nn.Module):
    """Simple encoder/decoder module for feature transformation."""

    def __init__(self, input_shape, output_shape):
        super(Coder, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(input_shape, max(input_shape, output_shape))
        self.fc2 = nn.Linear(max(input_shape, output_shape), output_shape)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)

    def forward(self, x):
        """Forward pass for feature transformation."""
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)


class CriticAttention(nn.Module):
    """Critic network with multi-head attention over agents.

    Computes Q-values using attention to model agent interactions.
    """

    def __init__(self, obs_shape_n, action_shape_n, args):
        super(CriticAttention, self).__init__()
        self.n_agents = len(obs_shape_n)
        self.feat_dim = args.critic_features_num  # Dimension of attention features

        # Encoders/decoders for each agent's obs+action
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(self.n_agents):
            self.encoders.append(Coder(obs_shape_n[i] + action_shape_n[i], self.feat_dim).to(args.device))
            self.decoders.append(Coder(self.feat_dim, 1).to(args.device))

        # Multi-head attention module
        self.attention = MultiHeadAttention(in_features=self.feat_dim, head_num=1)

        # Precompute observation/action slice ranges for each agent
        self.obs_ranges = []
        self.action_ranges = []
        obs_end, action_end = 0, 0
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            self.obs_ranges.append((obs_end, obs_end + obs_shape))
            self.action_ranges.append((action_end, action_end + action_shape))
            obs_end += obs_shape
            action_end += action_shape

    def forward(self, obs_input, action_input):
        """Forward pass to compute Q-values for all agents.

        Args:
            obs_input: Combined observations (batch_size, sum(obs_shape_n))
            action_input: Combined actions (batch_size, sum(action_shape_n))

        Returns:
            Q-values for each agent (batch_size, n_agents)
        """
        # Encode each agent's obs+action into features
        agent_features = []
        for i in range(self.n_agents):
            obs_slice = slice(self.obs_ranges[i][0], self.obs_ranges[i][1])
            action_slice = slice(self.action_ranges[i][0], self.action_ranges[i][1])
            agent_input = torch.cat([obs_input[:, obs_slice], action_input[:, action_slice]], dim=1)
            agent_features.append(self.encoders[i](agent_input))

        # Reshape for attention (batch_size, n_agents, feat_dim)
        features = torch.cat(agent_features, dim=1).reshape(-1, self.n_agents, self.feat_dim)

        # Apply multi-head attention
        attn_output = self.attention(features, features, features)

        # Decode attention outputs to Q-values
        q_values = []
        for i in range(self.n_agents):
            q_values.append(self.decoders[i](attn_output[:, i]))

        return torch.cat(q_values, dim=1)

    def attn_mat(self, obs_input, action_input):
        """Compute attention matrix (for analysis/visualization).

        Returns:
            Attention weights (batch_size, n_agents, n_agents)
        """
        # Encode agent features (same as forward pass)
        agent_features = []
        for i in range(self.n_agents):
            obs_slice = slice(self.obs_ranges[i][0], self.obs_ranges[i][1])
            act_slice = slice(self.action_ranges[i][0], self.action_ranges[i][1])
            agent_input = torch.cat([obs_input[:, obs_slice], action_input[:, act_slice]], dim=1)
            agent_features.append(self.encoders[i](agent_input))

        # Reshape for attention
        features = torch.stack(agent_features, dim=1)

        # Compute attention scores (reshape back to batch_size from batch_size * head_num)
        attn_scores = self.attention.scores(features, features, features)
        return attn_scores


class ActorAttention(nn.Module):
    """Actor network with multi-head attention over agent observations.

    Computes stochastic policy for each agent by attending to other agents' observations.
    Input: Concatenated observations of all agents
    Output: Stochastic policy (action probabilities) for the target agent
    """

    def __init__(self, obs_shape_n, action_shape_n, args):
        """
        Args:
            obs_shape_n: List of observation dimensions for each agent
            action_shape_n: List of action dimensions for each agent
            args: Training arguments (device, actor_features_num, head_num, actor_obs_size)
        """
        super(ActorAttention, self).__init__()
        self.feat_dim = args.actor_features_num  # Attention feature dimension
        self.head_num = args.head_num  # Number of attention heads
        self.device = args.device
        self.obs_ranges = args.actor_obs_size  # Precomputed observation slices for each agent
        self.action_dim = 5  # Action dimension (adjust if your env has different action space)

        # Encoders: project each agent's observation to feat_dim
        self.encoders = nn.ModuleList()
        for rng in self.obs_ranges:
            obs_dim = rng[1] - rng[0]
            self.encoders.append(Coder(obs_dim, self.feat_dim).to(self.device))

        # Multi-head attention over agent observations
        self.attention = MultiHeadAttention(
            in_features=self.feat_dim,
            head_num=self.head_num,
        )

        # Class token (aggregates attention outputs for policy prediction)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feat_dim, device=self.device), requires_grad=False)

        # Policy head: maps aggregated features to action logits
        self.fc_out = nn.Linear(self.feat_dim, self.action_dim).to(self.device)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize policy head weights (encoders/attention already initialized)."""
        nn.init.xavier_uniform_(self.fc_out.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, obs_input, model_original_out=False):
        """Forward pass to compute stochastic policy for agents.

        Args:
            obs_input: Combined observations (batch_size, sum(obs_shape_n))
            model_original_out: If True, return (logits, policy); else return policy

        Returns:
            If model_original_out: (logits, policy) (batch_size, action_dim)
            Else: policy (batch_size, action_dim)
        """
        # Add batch dimension if input is 1D (single sample)
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)

        batch_size = obs_input.size(0)

        # Encode each agent's observation to feature space
        agent_features = []
        for i in range(len(self.obs_ranges)):
            obs_slice = slice(self.obs_ranges[i][0], self.obs_ranges[i][1])
            agent_obs = obs_input[:, obs_slice]
            agent_features.append(self.encoders[i](agent_obs))

        # Reshape features for attention (batch_size, n_agents, feat_dim)
        features = torch.stack(agent_features, dim=1)

        # Add class token (batch_size, 1, feat_dim) to feature sequence
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        features_with_cls = torch.cat([cls_token, features], dim=1)  # (batch_size, n_agents + 1, feat_dim)

        # Apply multi-head self-attention
        attn_output = self.attention(q=features_with_cls, k=features_with_cls, v=features_with_cls)
        cls_output = attn_output[:, 0, :]

        logits = self.fc_out(cls_output)

        # Gumbel-softmax for stochastic policy
        u = torch.rand_like(logits)
        policy = F.softmax(logits -torch.log(-torch.log(u)), dim=-1)

        if model_original_out:
            return logits, policy
        return policy

    def attn_mat(self, obs_input):
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)

        # Encode each agent's observation to feature space
        agent_features = []
        for i in range(len(self.obs_ranges)):
            obs_slice = slice(self.obs_ranges[i][0], self.obs_ranges[i][1])
            agent_obs = obs_input[:, obs_slice]
            agent_features.append(self.encoders[i](agent_obs))

        # Reshape features for attention (batch_size, n_agents, feat_dim)
        features = torch.stack(agent_features, dim=1)

        # Add class token (batch_size, 1, feat_dim) to feature sequence
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=features.shape[0])
        features_with_cls = torch.cat([cls_token, features], dim=1)  # (batch_size, n_agents + 1, feat_dim)

        # Apply multi-head self-attention
        attn_scores = self.attention.scores(features_with_cls, features_with_cls, features_with_cls)
        return attn_scores
