import numpy as np
import os
import string
import torch


def log_validation(args, episode_infos):
    model_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
    f = open(os.path.join('valid', model_name), 'a')
    print('model_name: ', args.old_model_name, file=f)
    print('test_episode: ', args.test_episode, file=f)
    print('fault_time: ', args.fault_time, file=f)
    test_info = {}

    for episode_info in episode_infos:
        for k, v in episode_info.items():
            if k not in test_info.keys():
                test_info[k] = 0
            test_info[k] += v
    if args.env == 'fix':
        success = 0
        for episode_info in episode_infos:
            if ('times_fix' not in episode_info.keys() or episode_info['times_fix'] == 1) and episode_info['times_colli'] == 1:
                success += 1
        print('success: ', success / args.test_episode, file=f)
    if args.env == 'spread':
        success = 0
        for episode_info in episode_infos:
            if episode_info['occupied'] == 2:
                success += 1
        print('success: ', success / args.test_episode, file=f)
    for k, v in test_info.items():
        print('{}: {}'.format(k, v / args.test_episode), file=f)


def local_print(episode_metrics_list):
    metrics2print = {}
    metrics2print_mean = {}
    for metrics in episode_metrics_list[-200:]:
        for k, v in metrics.items():
            if k not in metrics2print.keys():
                metrics2print[k] = []
            metrics2print[k].append(v)
    for k, v in metrics2print.items():
        if isinstance(v[0], torch.Tensor):
            # metrics2print_mean[k] = [torch.mean(torch.Tensor(v))]
            pass
        else:
            metrics2print_mean[k] = [round(np.mean(v), 2)]
    print('\r=Training: episode:{}'.format(len(episode_metrics_list)), end='')

    print(" " * 20, end='')
    for k, v in metrics2print_mean.items():
        print(k + ': ' + str(v) + '  ', end='')


def episode_info2metrics(episode_info, step_infos):
    for step_info in step_infos:
        normal_bool_list = ~np.array(step_info['fault_info']['fault_list'])
        num_normal_agents = np.sum(normal_bool_list)
        episode_info['rew'] += np.sum(np.array(step_info['rew'])[normal_bool_list]) / num_normal_agents
    return episode_info


class AttPrint:
    def __init__(self, alg_controller, args):
        self.alg_controller = alg_controller
        self.critic_att_pre = []
        self.critic_att_post = []
        self.actor_att_pre = [[] for i in range(args.num_adversaries)]
        self.actor_att_post = [[] for i in range(args.num_adversaries)]
        self.args = args

    def add_critic_att(self, obs_n, action_n, fault_info):
        import torch
        obs_list = torch.from_numpy(np.concatenate(obs_n).reshape(1, -1)).to(self.args.device, torch.float)
        action_list = torch.from_numpy(np.concatenate(action_n).reshape(1, -1)).to(self.args.device, torch.float)
        attentions_mat = self.alg_controller.critics_cur[0].attn_mat(obs_list, action_list).detach().cpu().numpy()
        attentions_mat = np.mean(attentions_mat, axis=0)
        if not any(fault_info['fault_list']):
            self.critic_att_pre.append(attentions_mat)
        elif not fault_info['fault_change']:
            self.critic_att_post.append(attentions_mat)

    def print_critic_att(self):
        print('critic_att_pre: ', np.mean(np.array(self.critic_att_pre), axis=0))
        print('critic_att_post: ', np.mean(np.array(self.critic_att_post), axis=0))

    def add_actor_att(self, obs_n, fault_info):
        import torch
        for i in range(self.args.num_adversaries):
            if not any(fault_info['fault_list']):
                attentions_mat = self.alg_controller.actors_cur[i].attn_mat(
                    torch.from_numpy(obs_n[i]).to(self.args.device, torch.float)).detach().cpu().numpy()[:, 0]
                self.actor_att_pre[i].append(np.mean(attentions_mat, axis=0))
            elif not fault_info['fault_change']:
                attentions_mat = self.alg_controller.actors_cur[i].attn_mat(
                    torch.from_numpy(obs_n[i]).to(self.args.device, torch.float)).detach().cpu().numpy()[:, 0]
                self.actor_att_post[i].append(np.mean(attentions_mat, axis=0))

    def print_actor_att(self):
        print('actor_att_pre: ', np.mean(np.array(self.actor_att_pre), axis=1))
        print('actor_att_post: ', np.mean(np.array(self.actor_att_post), axis=1))

