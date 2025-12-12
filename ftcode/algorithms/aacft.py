import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ftcode.algorithms.maddpg import MADDPGLearner
from ftcode.algorithms.module_utils import CriticAttention, ActorAttention


class AACFTLearner(MADDPGLearner):
    """Attention-Augmented Critic and/or Actor for Fault-Tolerant MADDPG.

    Extends MADDPG with attention mechanisms in either actor, critic, or both networks
    to better handle multi-agent interactions under fault conditions.
    """

    def __init__(self, args, env_args, ex_name, actor_attention, critic_attention):
        """Initialize AACFTLearner.

        Args:
            args: Configuration parameters
            env_args: Environment parameters (observation shapes, action shapes)
            ex_name: Experiment name for logging/saving
            actor_attention: Whether to use attention in actor networks
            critic_attention: Whether to use attention in critic networks
        """
        super().__init__(args, env_args, ex_name)
        self.actor_attention = actor_attention
        self.critic_attention = critic_attention
        self.flag = args.flag  # Flag value for modifying faulty observations
        # self.offset = [[4, 8], [11, 15]]  # Observation offsets for fault masking (domain-specific)

        # Initialize attention-based actor networks if enabled
        if actor_attention:
            if args.ps:  # Parameter sharing across agents
                assert len(set(self.obs_shape_n)) == 1, "All agents must share observation space for PS."
                assert len(set(self.action_shape_n)) == 1, "All agents must share action space for PS."
                obs_shape = self.obs_shape_n[0]
                action_shape = self.action_shape_n[0]
                self.actors_cur = [ActorAttention(obs_shape, action_shape, args).to(args.device)]
                self.actors_tar = [ActorAttention(obs_shape, action_shape, args).to(args.device)]
                self.optimizers_a = [optim.Adam(self.actors_cur[0].parameters(), args.lr_a)]
            else:  # Independent actors
                self.actors_cur = [
                    ActorAttention(self.obs_shape_n[i], self.action_shape_n[i], args).to(args.device)
                    for i in range(self.n_agents)
                ]
                self.actors_tar = [
                    ActorAttention(self.obs_shape_n[i], self.action_shape_n[i], args).to(args.device)
                    for i in range(self.n_agents)
                ]
                self.optimizers_a = [
                    optim.Adam(self.actors_cur[i].parameters(), args.lr_a)
                    for i in range(self.n_agents)
                ]

        # Initialize attention-based critic networks if enabled
        if critic_attention:
            self.critics_cur = [CriticAttention(self.obs_shape_n, self.action_shape_n, args).to(args.device)]
            self.critics_tar = [CriticAttention(self.obs_shape_n, self.action_shape_n, args).to(args.device)]
            self.optimizers_c = [optim.Adam(self.critics_cur[0].parameters(), args.lr_c)]

        # Sync target networks with current networks initially
        self.update_target_networks(1.0)

        # Metrics tracking
        self.fault_cnt_per_update = 0
        self.fault_time = 0
        self.nofault_time = 0

    def obs_fault_modify(self, obs, fault_list):
        """Modify observations to reflect agent faults (for attention mechanisms).

        Args:
            obs: List of observations for each agent
            fault_list: List indicating which agents are faulty (bool)
        """
        if self.flag is None:
            return

        # Mask observations of faulty agents if critic uses attention
        if self.critic_attention:
            for fault_id, is_faulty in enumerate(fault_list):
                if is_faulty:
                    obs[fault_id] = self.flag  # Replace with flag value (e.g., placeholder)

        # if self.actor_attention:
        #     for fault_id, fault_bool in enumerate(fault_list):
        #         if fault_bool:
        #             for i in range(self.n_agents):
        #                 if not fault_id == i:
        #                     obs[i][self.offset[fault_id - (fault_id > i)][0]: self.offset[fault_id - (fault_id > i)][1]] = self.flag

    def alg_info2metrics(self, episode_info):
        """Log fault-related metrics to episode information.

        Args:
            episode_info: Dictionary to store episode metrics
        """
        episode_info['batch_fault_cnt'] = self.fault_cnt_per_update / self.args.batch_size

        # Track fault statistics in replay buffer
        fault_cnt = 0
        memory_fault_time = np.zeros(40, dtype=int)
        memory_nofault_time = np.zeros(30, dtype=int)

        for entry in self.memory._storage:
            fault_bool = np.sum(entry[5]['fault_list'])
            if fault_bool > 0:
                fault_cnt += 1
                memory_fault_time[entry[5]['current_time'] - entry[5]['fault_time']] += 1
            else:
                memory_nofault_time[entry[5]['current_time']] += 1

        episode_info['memory_fault_cnt'] = fault_cnt / len(self.memory._storage)
        episode_info['batch_fault_time'] = self.fault_time
        episode_info['batch_nofault_time'] = self.nofault_time
        episode_info['memory_fault_time'] = memory_fault_time
        episode_info['memory_nofault_time'] = memory_nofault_time

    def update(self, args, episode_id):
        """Perform one training iteration with attention mechanisms.

        Args:
            args: Configuration parameters
            episode_id: Current training episode number
        """
        if self.critic_attention:
            # Handle parameter sharing for actors
            actors_cur = self.actors_cur * self.n_agents if self.args.ps else self.actors_cur
            actors_tar = self.actors_tar * self.n_agents if self.args.ps else self.actors_tar
            optimizers_a = self.optimizers_a * self.n_agents if self.args.ps else self.optimizers_a

            # Sample batch from replay buffer
            batch_data = self.memory.sample(self.args.batch_size)
            obs_old, actions, rewards, obs_new, dones, fault_info = batch_data

            # Convert data to tensors
            device = self.args.device
            rewards = torch.from_numpy(rewards).to(device=device, dtype=torch.float)
            dones = torch.from_numpy(~dones).to(device=device, dtype=torch.float)
            actions = torch.from_numpy(actions).to(device=device, dtype=torch.float)
            obs_old = torch.from_numpy(obs_old).to(device=device, dtype=torch.float)
            obs_new = torch.from_numpy(obs_new).to(device=device, dtype=torch.float)

            # Identify non-faulty agents in the batch
            normal_mask = [
                (np.array(fault_info['fault_list'], dtype=bool)[:, agent_idx] == False)
                for agent_idx in range(self.n_agents)
            ]

            # Compute target actions using target actors
            target_actions = torch.cat([
                actor_tar(obs_new[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach()
                for idx, actor_tar in enumerate(actors_tar)
            ], dim=1)

            # Mask actions of faulty agents in target actions
            for agent_idx in range(self.n_agents):
                target_actions[~normal_mask[agent_idx],
                self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = 0

            # Update critic network
            normal_mask_tensor = torch.from_numpy(~np.array(fault_info['fault_list'], dtype=bool)).to(device)
            q_values = self.critics_cur[0](obs_old, actions)
            q_targets = self.critics_tar[0](obs_new, target_actions)
            target_values = q_targets * args.gamma * dones.reshape(self.args.batch_size, 1) + rewards

            # Compute TD error with masking for faulty agents
            td_error = (q_values - target_values).pow(2)
            td_error = td_error.masked_fill(~normal_mask_tensor, 0)  # Ignore faulty entries
            td_error = torch.sum(td_error, dim=1) / torch.sum(normal_mask_tensor, dim=1)
            loss_critic = td_error.mean()

            # Optimize critic
            self.optimizers_c[0].zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.critics_cur[0].parameters(), self.args.max_grad_norm)
            self.optimizers_c[0].step()

            # Update actor networks
            for agent_idx, (actor_cur, actor_tar, opt_a) in enumerate(zip(actors_cur, actors_tar, optimizers_a)):
                # Sample batch for current actor
                batch_data = self.memory.sample(self.args.batch_size)
                obs_old, actions, _, _, _, fault_info = batch_data

                # Filter non-faulty experiences
                normal_mask = (np.array(fault_info['fault_list'], dtype=bool)[:, agent_idx] == False)
                actions_filtered = torch.from_numpy(actions).to(device, torch.float)[normal_mask]
                obs_old_filtered = torch.from_numpy(obs_old).to(device, torch.float)[normal_mask]

                # Compute new policy for current agent
                _, new_policy = actor_cur(
                    obs_old_filtered[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]],
                    model_original_out=True
                )

                # Replace old action with new policy in action batch
                actions_filtered[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = new_policy

                # Optimize actor (maximize critic value)
                loss_actor = -torch.mean(self.critics_cur[0](obs_old_filtered, actions_filtered)[:, agent_idx])
                opt_a.zero_grad()
                loss_actor.backward()
                nn.utils.clip_grad_norm_(actor_cur.parameters(), self.args.max_grad_norm)
                opt_a.step()

            # Soft-update target networks
            self.update_target_networks(args.tao)

        else:
            # Fallback to base MADDPG update if no critic attention
            super().update(args, episode_id)
