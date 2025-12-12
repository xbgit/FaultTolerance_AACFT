import os.path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ftcode.algorithms.alg_controller import AlgController
from ftcode.algorithms.module_utils import ActorMLP, CriticMLP
from ftcode.utils.timer import timer


class MADDPGLearner(AlgController):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG) learner.

    Handles training of multiple agents with centralized critics and decentralized actors.
    Supports parameter sharing (PS) across agents when enabled.
    """

    def __init__(self, args, env_args, ex_name):
        super().__init__(args, env_args, ex_name)
        self.model_file_dir = os.path.join('models', ex_name)

        # Initialize actor networks (current and target)
        if args.ps:
            # Parameter sharing: all agents use the same actor
            assert len(set(self.obs_shape_n)) == 1, "All agents must have the same observation space for PS."
            assert len(set(self.action_shape_n)) == 1, "All agents must have the same action space for PS."
            obs_shape = self.obs_shape_n[0]
            action_shape = self.action_shape_n[0]
            self.actors_cur = [ActorMLP(obs_shape, action_shape, args.mlp_hidden_size).to(args.device)]
            self.actors_tar = [ActorMLP(obs_shape, action_shape, args.mlp_hidden_size).to(args.device)]
            self.optimizers_a = [optim.Adam(self.actors_cur[0].parameters(), args.lr_a)]
        else:
            # Independent actors for each agent
            self.actors_cur = [
                ActorMLP(self.obs_shape_n[i], self.action_shape_n[i], args.mlp_hidden_size).to(args.device)
                for i in range(self.n_agents)
            ]
            self.actors_tar = [
                ActorMLP(self.obs_shape_n[i], self.action_shape_n[i], args.mlp_hidden_size).to(args.device)
                for i in range(self.n_agents)
            ]
            self.optimizers_a = [
                optim.Adam(self.actors_cur[i].parameters(), args.lr_a)
                for i in range(self.n_agents)
            ]

        # Initialize critic networks (current and target)
        self.critics_cur = [
            CriticMLP(sum(self.obs_shape_n), sum(self.action_shape_n), args.mlp_hidden_size).to(args.device)
            for i in range(self.n_agents)
        ]
        self.critics_tar = [
            CriticMLP(sum(self.obs_shape_n), sum(self.action_shape_n), args.mlp_hidden_size).to(args.device)
            for i in range(self.n_agents)
        ]
        self.optimizers_c = [
            optim.Adam(self.critics_cur[i].parameters(), args.lr_c)
            for i in range(self.n_agents)
        ]

        # Sync target networks with current networks initially
        self.update_target_networks(1.0)

    def get_models(self):
        """Return all neural network models (actors + critics, current + target)."""
        return self.actors_cur + self.actors_tar + self.critics_cur + self.critics_tar

    def update_target_networks(self, tao):
        """Soft-update target networks using current networks.

        Args:
            tao: Interpolation factor (tao * current + (1-tao) * target)
        """
        agents_cur = self.actors_cur + self.critics_cur
        agents_tar = self.actors_tar + self.critics_tar

        with torch.no_grad():  # No gradient computation for target updates
            for curr_net, target_net in zip(agents_cur, agents_tar):
                for curr_param, target_param in zip(curr_net.parameters(), target_net.parameters()):
                    target_param.data.mul_(1 - tao)
                    target_param.data.add_(tao * curr_param.data)

    def update(self, args, episode_id):
        """Perform one training iteration (update critics and actors)."""
        # Handle parameter sharing: replicate actors if needed
        actors_cur = self.actors_cur * self.n_agents if self.args.ps else self.actors_cur
        actors_tar = self.actors_tar * self.n_agents if self.args.ps else self.actors_tar
        optimizers_a = self.optimizers_a * self.n_agents if self.args.ps else self.optimizers_a

        # Update each agent's network
        for agent_idx, (actor_cur, actor_tar, critic_cur, critic_tar, opt_a, opt_c) in enumerate(zip(
                actors_cur, actors_tar, self.critics_cur, self.critics_tar, optimizers_a, self.optimizers_c)):

            # Sample batch data and convert to tensors
            batch_data = self.memory.sample(self.args.batch_size)
            obs_old, actions, rewards, obs_new, dones, fault_info = batch_data

            # Identify non-faulty agents in the batch
            normal_mask = [
                (np.array(fault_info['fault_list'], dtype=bool)[:, agent_idx] == False)
                for agent_idx in range(self.n_agents)
            ]

            # Convert data to tensors
            device = self.args.device
            rewards = torch.from_numpy(rewards).to(device=device, dtype=torch.float)[normal_mask[agent_idx]]
            dones = torch.from_numpy(~dones).to(device=device, dtype=torch.float)[normal_mask[agent_idx]]
            actions = torch.from_numpy(actions).to(device=device, dtype=torch.float)[normal_mask[agent_idx]]
            obs_old = torch.from_numpy(obs_old).to(device=device, dtype=torch.float)[normal_mask[agent_idx]]
            obs_new = torch.from_numpy(obs_new).to(device=device, dtype=torch.float)[normal_mask[agent_idx]]

            # Compute target actions using target actors
            target_actions = torch.cat([
                actor_tar(obs_new[:, self.obs_size[idx][0]:self.obs_size[idx][1]]).detach()
                for idx, actor_tar in enumerate(actors_tar)
            ], dim=1)

            # Mask actions for faulty agents
            for idx in range(self.n_agents):
                target_actions[~normal_mask[idx][normal_mask[agent_idx]],
                self.action_size[idx][0]:self.action_size[idx][1]] = 0

            # Update critic
            q_values = critic_cur(obs_old, actions).reshape(-1)
            q_targets = critic_tar(obs_new, target_actions).reshape(-1)
            target_values = q_targets * self.args.gamma * dones + rewards[:, agent_idx]

            # Critic loss (MSE between predicted and target Q-values)
            loss_critic = nn.MSELoss()(q_values, target_values.detach())
            opt_c.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(critic_cur.parameters(), self.args.max_grad_norm)  # Prevent gradient explosion
            opt_c.step()

            # Update actor (using deterministic policy gradient)
            _, new_policy = actor_cur(
                obs_old[:, self.obs_size[agent_idx][0]:self.obs_size[agent_idx][1]],
                model_original_out=True
            )
            # Replace old action with new policy for the current agent
            actions[:, self.action_size[agent_idx][0]:self.action_size[agent_idx][1]] = new_policy

            # Actor loss (maximize Q-value, hence negative mean Q)
            loss_actor = -torch.mean(critic_cur(obs_old, actions))
            opt_a.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(actor_cur.parameters(), self.args.max_grad_norm)
            opt_a.step()

        # Soft-update target networks after all agents are updated
        self.update_target_networks(args.tao)

        # Clear GPU cache if using CUDA
        if args.device != 'cpu':
            torch.cuda.empty_cache()

    def policy(self, obs_n, fault_controller, fault_info, training_mode=True):
        """Generate actions for all agents using current policies.

        Args:
            obs_n: List of observations for each agent
            fault_controller: Fault management controller (not used here)
            fault_info: Dictionary with fault status info
            training_mode: Whether to use stochastic (train) or deterministic (eval) policy

        Returns:
            List of actions for each agent
        """
        actors_cur = self.actors_cur * self.n_agents if self.args.ps else self.actors_cur

        if training_mode:
            # Stochastic policy (used during training)
            action_n = [
                actor(obs.to(self.args.device, torch.float)).detach().cpu().numpy()
                for actor, obs in zip(actors_cur, obs_n)
            ]
        else:
            # Deterministic policy (used during evaluation)
            action_n = []
            for actor, obs in zip(actors_cur, obs_n):
                model_out, _ = actor(obs.to(self.args.device, torch.float), model_original_out=True)
                action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())

        return action_n

    def save_model(self, episode_id):
        """Save current model weights to disk.

        Args:
            episode_id: Current training episode (used for file naming)
        """
        path = os.path.join(self.model_file_dir, str(episode_id))
        os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist

        # Save actor networks
        for agent_idx, (actor_cur, actor_tar) in enumerate(zip(self.actors_cur, self.actors_tar)):
            torch.save(actor_cur, os.path.join(path, f'a_c_{agent_idx}.pt'))
            torch.save(actor_tar, os.path.join(path, f'a_t_{agent_idx}.pt'))

        # Save critic networks
        for agent_idx, (critic_cur, critic_tar) in enumerate(zip(self.critics_cur, self.critics_tar)):
            torch.save(critic_cur, os.path.join(path, f'c_c_{agent_idx}.pt'))
            torch.save(critic_tar, os.path.join(path, f'c_t_{agent_idx}.pt'))

    def load_model(self, episode_id):
        """Load model weights from disk.

        Args:
            episode_id: Episode to load weights from
        """
        path = os.path.join(self.model_file_dir, str(episode_id))
        n_actor = len(self.actors_cur)
        n_critic = len(self.critics_cur)

        # Load actor networks
        self.actors_cur, self.actors_tar = [], []
        for i in range(n_actor):
            self.actors_cur.append(torch.load(os.path.join(path, f'a_c_{i}.pt'), map_location=self.args.device))
            self.actors_tar.append(torch.load(os.path.join(path, f'a_t_{i}.pt'), map_location=self.args.device))

        # Load critic networks
        self.critics_cur, self.critics_tar = [], []
        for i in range(n_critic):
            self.critics_cur.append(torch.load(os.path.join(path, f'c_c_{i}.pt'), map_location=self.args.device))
            self.critics_tar.append(torch.load(os.path.join(path, f'c_t_{i}.pt'), map_location=self.args.device))

    def save_all(self, episode_id, episode_metrics_list):
        """Save full training state (models, optimizers, memory, metrics)."""
        os.makedirs(self.model_file_dir, exist_ok=True)

        # Save replay memory
        with open(os.path.join(self.model_file_dir, 'memory.pkl'), "wb") as f:
            pickle.dump(self.memory, f)

        # Save training metrics
        with open(os.path.join(self.model_file_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(episode_metrics_list, f)

        # Save optimizers
        for agent_idx, opt_a in enumerate(self.optimizers_a):
            torch.save(opt_a, os.path.join(self.model_file_dir, f'opt_a_{agent_idx}.pt'))
        for agent_idx, opt_c in enumerate(self.optimizers_c):
            torch.save(opt_c, os.path.join(self.model_file_dir, f'opt_c_{agent_idx}.pt'))

        # Save model weights
        self.save_model(episode_id)

    def load_all(self, episode_id):
        """Load full training state (models, optimizers, memory)."""
        # Load replay memory
        with open(os.path.join(self.model_file_dir, 'memory.pickle'), "rb") as f:
            self.memory = pickle.load(f)

        # Load optimizers
        self.optimizers_a, self.optimizers_c = [], []
        for agent_idx in range(len(self.optimizers_a)):
            self.optimizers_a.append(torch.load(os.path.join(self.model_file_dir, f'opt_a_{agent_idx}.pt')))
        for agent_idx in range(len(self.optimizers_c)):
            self.optimizers_c.append(torch.load(os.path.join(self.model_file_dir, f'opt_c_{agent_idx}.pt')))

        # Load model weights
        self.load_model(episode_id)

    def prep_training(self, device):
        """Prepare networks for training (set to train mode, move to device)."""
        for net in self.get_models():
            net.train()
            net.to(device)

    def prep_rollouts(self, device):
        """Prepare networks for rollouts (set to eval mode, move to device)."""
        for net in self.get_models():
            net.eval()
            net.to(device)