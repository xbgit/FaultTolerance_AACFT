from ftcode.algorithms.maddpg import MADDPGLearner
import torch
import torch.nn as nn


class M3DDPGLearner(MADDPGLearner):
    def update(self, args):
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in enumerate(zip(
                self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar, self.optimizers_a, self.optimizers_c)):
            device = self.args.device
            batch_data = self.memory.sample(self.args.batch_size)
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done, fault_info = batch_data
            rew = torch.tensor(_rew_n, device=device, dtype=torch.float)
            done = torch.tensor(~_done, dtype=torch.float, device=device)
            action_cur_o = torch.from_numpy(_action_n).to(device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(device, torch.float)

            action_tar = torch.cat([a_t(obs_n_n[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach()
                                    for idx, a_t in enumerate(self.actors_tar)], dim=1)
            action_tar = torch.tensor(action_tar.detach().numpy(), device=args.device, dtype=torch.float32,
                                      requires_grad=True)
            target_loss = -critic_t(obs_n_n, action_tar).mean()
            target_loss.backward()

            gradients = action_tar.grad.detach()
            gradients = 0.01 * gradients
            gradients[:, self.action_size[agent_idx][0]: self.action_size[agent_idx][1]] = 0
            action_tar = action_tar.detach() + gradients

            q = critic_c(obs_n_o, action_cur_o).reshape(-1)
            q_ = critic_t(obs_n_n, action_tar).reshape(-1)
            tar_value = q_ * self.args.gamma * done + rew[:, agent_idx]
            loss_c = torch.nn.MSELoss()(q, tar_value)
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), self.args.max_grad_norm)
            opt_c.step()

            model_out, policy_c_new = actor_c(obs_n_o[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]],
                                              model_original_out=True)

            action_cur_o[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = policy_c_new
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), self.args.max_grad_norm)
            opt_a.step()

        self.update_target_networks(args.tao)

