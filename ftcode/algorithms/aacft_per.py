import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat
from ftcode.algorithms.maddpg import MADDPGLearner
from ftcode.algorithms.module_utils import Coder, MultiHeadAttention
from ftcode.replay_buffer import ReplayBufferPrioritized

class AACFTPERLearner(MADDPGLearner):
    def __init__(self, args, env_args, ex_name, actor_attention, critic_attention):
        super().__init__(args, env_args, ex_name)
        self.memory = ReplayBufferPrioritized(args.memory_size, args.batch_size, self.n_agents)
        self.actor_attention = actor_attention
        self.critic_attention = critic_attention
        self.offset = [[4, 8], [11, 15]]
        if actor_attention:
            self.actors_cur = [ActorAttention(self.obs_shape_n[i], self.action_shape_n[i], args).
                               to(args.device) for i in range(self.n_agents)]
            self.actors_tar = [ActorAttention(self.obs_shape_n[i], self.action_shape_n[i], args).
                               to(args.device) for i in range(self.n_agents)]
            self.optimizers_a = [optim.Adam(self.actors_cur[i].parameters(), args.lr_a) for i in range(self.n_agents)]
        if critic_attention:
            self.critics_cur = [CriticAttention(self.obs_shape_n, self.action_shape_n, args).to(args.device)]
            self.critics_tar = [CriticAttention(self.obs_shape_n, self.action_shape_n, args).to(args.device)]
            self.optimizers_c = [optim.Adam(self.critics_cur[0].parameters(), args.lr_a)]

        self.fault_cnt_per_update = 0

    def obs_fault_modify(self, obs, fault_list):
        if self.critic_attention:
            for fault_id, fault_bool in enumerate(fault_list):
                if fault_bool:
                    obs[fault_id][:] = 10
        if self.actor_attention:
            for fault_id, fault_bool in enumerate(fault_list):
                if fault_bool:
                    for i in range(self.n_agents):
                        if not fault_id == i:
                            obs[i][self.offset[fault_id - (fault_id > i)][0]: self.offset[fault_id - (fault_id > i)][1]] = 10

    def alg_info2metrics(self, episode_info):
        episode_info['batch_fault_cnt'] = self.fault_cnt_per_update / self.args.batch_size
        fault_cnt = 0
        for v in self.memory._storage.values():
            fault_bool = np.sum(v[5]['fault_list'])
            if fault_bool > 0:
                fault_cnt += 1
        episode_info['memory_fault_cnt'] = fault_cnt / len(self.memory._storage)

    def update(self, args, episode_id):
        if episode_id % 100 == 0:
            self.memory.rebalance()
        if self.critic_attention:
            beta = min(1.0, 0.4 + episode_id * 0.6 / args.max_episode)
            device = self.args.device
            batch_data = self.memory.sample(beta, -1)
            data, weights, sample_indices = batch_data
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done, fault_info = data
            rew = torch.tensor(_rew_n, device=device, dtype=torch.float)
            done = torch.tensor(~_done, dtype=torch.float, device=device)
            action_cur_o = torch.from_numpy(_action_n).to(device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(device, torch.float)
            weights = torch.from_numpy(weights).to(device, torch.float)

            normal_list = torch.from_numpy(~np.array(fault_info['fault_list'])).to(device)
            self.fault_cnt_per_update = torch.sum(~normal_list)

            action_tar = torch.cat([a_t(obs_n_n[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach()
                                    for idx, a_t in enumerate(self.actors_tar)], dim=1)

            for agent_idx in range(self.n_agents):
                action_tar[~normal_list[:, agent_idx], self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = 0

            q = self.critics_cur[0](obs_n_o, action_cur_o)
            q_ = self.critics_tar[0](obs_n_n, action_tar)
            tar_value = torch.mul(q_ * args.gamma, done.reshape(self.args.batch_size, 1)) + rew
            td_error = (q - tar_value).pow(2)
            td_error.masked_fill(~normal_list, 0)
            td_error = torch.sum(td_error, dim=1) / torch.sum(normal_list, dim=1)
            weighted_td_error = weights * td_error
            loss_c = weighted_td_error.mean()

            priorities = td_error.data.cpu().numpy()
            self.memory.update_priority(sample_indices, priorities, -1)

            self.optimizers_c[0].zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critics_cur[0].parameters(), self.args.max_grad_norm)
            self.optimizers_c[0].step()

            for agent_idx, (actor_c, actor_t, opt_a) in enumerate(zip(self.actors_cur, self.actors_tar, self.optimizers_a)):
                device = self.args.device
                data, weights, sample_indices = self.memory.sample(beta, agent_idx)
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done, fault_info = data
                normal_list = torch.from_numpy(~np.array(fault_info['fault_list']))[:, agent_idx]
                _obs_n_o, _action_n, _rew_n, _obs_n_n, _done, fault_info = data
                action_cur_o = torch.from_numpy(_action_n).to(device, torch.float)[normal_list]
                obs_n_o = torch.from_numpy(_obs_n_o).to(device, torch.float)[normal_list]
                weights = torch.from_numpy(weights).to(device, torch.float)[normal_list]

                model_out, policy_c_new = actor_c(obs_n_o[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]],
                                                  model_original_out=True)
                action_cur_o[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = policy_c_new
                td_error = torch.mul(-1, self.critics_cur[0](obs_n_o, action_cur_o)[:, agent_idx])
                weighted_td_error = weights * td_error
                loss_a = weighted_td_error.mean()

                priorities = td_error.data.cpu().numpy()
                self.memory.update_priority(np.array(sample_indices)[normal_list], priorities, agent_idx)

                opt_a.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor_c.parameters(), self.args.max_grad_norm)
                opt_a.step()

        else:
            super().update(args, episode_id)

        self.update_target_networks(args.tao)


class CriticAttention(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(CriticAttention, self).__init__()
        self.agents_n = len(obs_shape_n)
        self.features_n = args.critic_features_num
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        for i in range(self.agents_n):
            self.encoder.append(Coder(obs_shape_n[i] + action_shape_n[i], self.features_n).to(args.device))
            self.decoder.append(Coder(self.features_n, 1).to(args.device))

        self.attention = MultiHeadAttention(in_features=self.features_n, head_num=1)
        self.obs_size = []
        self.action_size = []
        head_o, head_a, end_o, end_a = 0, 0, 0, 0
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            end_o = end_o + obs_shape
            end_a = end_a + action_shape
            range_o = (head_o, end_o)
            range_a = (head_a, end_a)
            self.obs_size.append(range_o)
            self.action_size.append(range_a)
            head_o = end_o
            head_a = end_a

    def forward(self, obs_input, action_input):
        f_ = []
        for i in range(self.agents_n):
            t = self.encoder[i](torch.cat([obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]], action_input[:, self.action_size[i][0]:self.action_size[i][1]]], dim=1))
            f_.append(t)
        f = torch.cat(f_, dim=1).reshape(-1, self.agents_n, self.features_n)

        values = self.attention(f, f, f)
        out = []
        for i in range(self.agents_n):
            out.append(self.decoder[i](values[:, i]))
        return torch.cat(out, dim=1)

    def attn_mat(self, obs_input, action_input):
        f_ = []
        for i in range(self.agents_n):
            t = self.encoder[i](torch.cat([obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]], action_input[:, self.action_size[i][0]:self.action_size[i][1]]], dim=1))
            f_.append(t)
        f = torch.cat(f_, dim=1).reshape(-1, self.agents_n, self.features_n)

        return self.attention.scores(f, f, f)


class ActorAttention(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(ActorAttention, self).__init__()
        self.features_n = args.actor_features_num

        self.obs_size = [[0, 4], [4, 11], [11, 18], [18, 22], [22, 26]]
        self.encoder = nn.ModuleList([Coder((item[1] - item[0]), self.features_n).to(args.device) for item in self.obs_size])

        self.attention = MultiHeadAttention(in_features=self.features_n, head_num=args.head_num)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.features_n), requires_grad=False)

        self.linear_head = nn.Linear(self.features_n, 5)

    def forward(self, obs_input, model_original_out=False):
        f_ = []
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)
        for i in range(len(self.obs_size)):
            f_.append(self.encoder[i](obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]]))

        f = torch.cat(f_, dim=1).reshape(-1, len(self.obs_size), self.features_n)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=f.shape[0]).clone()
        f = torch.cat((cls_tokens, f), dim=1)

        x = self.attention(f, f, f)
        x = x[:, 0]
        model_out = self.linear_head(x).squeeze(0)

        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:  return model_out, policy # for model_out criterion

        return policy

    def attn_mat(self, obs_input):
        f_ = []
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)
        for i in range(len(self.obs_size)):
            f_.append(self.encoder[i](obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]]))
        x = torch.cat(f_, dim=1).reshape(-1, len(self.obs_size), self.features_n)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]).clone()
        x = torch.cat((cls_tokens, x), dim=1)
        return self.attention.scores(x, x, x)

