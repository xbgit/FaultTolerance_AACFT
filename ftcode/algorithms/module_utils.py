import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CriticMLP(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, hidden_size):
        super(CriticMLP, self).__init__()
        self.obs_shape_n = obs_shape_n
        self.action_shape_n = action_shape_n
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, hidden_size)
        self.linear_c2 = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(
            torch.cat([obs_input[:, 0: self.obs_shape_n], action_input[:, 0: self.action_shape_n]], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class ActorMLP(nn.Module):
    def __init__(self, num_inputs, action_size, hidden_size):
        super(ActorMLP, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, hidden_size)
        self.linear_a2 = nn.Linear(hidden_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, action_size)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:   return model_out, policy  # for model_out criterion
        return policy


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_q.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_k.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_v.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_o.weight, mean=0, std=0.1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def scores(self, q, k, v):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)
        attentions = F.softmax(scores, dim=-1)
        return attentions

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class Coder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Coder, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(input_shape, 64)
        self.linear_c2 = nn.Linear(64, output_shape)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_c1.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_c2.weight, mean=0, std=0.1)

    def forward(self, x):
        x = self.LReLU(self.linear_c1(x))
        x = self.linear_c2(x)
        return x

