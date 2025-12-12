import argparse
import torch
import time
import os
import numpy as np
from torch.autograd import Variable
from domain import make_env
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from ftcode.curriculums.curriculum import make_cl
from ftcode.logger import local_print, episode_info2metrics, AttentionPrinter, log_validation
import wandb


def make_parallel_env(domain_name, scenario_name, n_rollout_threads, args, cl_controller):
    """
    Create parallel environments using vectorized wrappers.

    Args:
        domain_name: Name of the environment domain
        scenario_name: Name of the specific scenario
        n_rollout_threads: Number of parallel rollout threads
        args: Configuration arguments
        cl_controller: Curriculum learning controller

    Returns:
        Vectorized environment (DummyVecEnv for 1 thread, SubprocVecEnv for multiple)
    """

    def get_env_fn(rank):
        def init_env():
            env, _ = make_env(domain_name, scenario_name, args, cl_controller)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run_train(alg_controller, fault_controller, start_episode, args):
    """
    Main training loop for multi-agent reinforcement learning.

    Args:
        alg_controller: Controller for the reinforcement learning algorithm
        fault_controller: Controller for managing fault injection
        start_episode: Initial episode number (for resuming training)
        args: Configuration arguments
    """
    training_mode = True  # Enable stochastic policies during training

    # Initialize environments and controllers
    cl_controller = make_cl(args.cl, args)
    env = make_parallel_env(args.domain, args.env, args.n_rollout_threads, args, cl_controller)
    tmp_env, kwargs = make_env(args.domain, args.env, args, cl_controller)  # Temp env for metadata

    # Initialize training counters and buffers
    step_id = 0
    update_id = 0
    save_id = 1
    episode_id = start_episode + 1
    obs_n, fault_info = env.reset()

    episode_metrics_list = []
    episode_buffers = [{
        'step_infos': [None] * args.per_episode_max_len,
        'current_idx': 0,
        'episode_info': {'rew': 0}
    } for _ in range(args.n_rollout_threads)]

    # Main training loop
    while episode_id <= args.max_episode:
        # Prepare algorithm for rollouts
        alg_controller.prep_rollouts(device=args.device)

        # Convert observations to torch Variables
        torch_obs = [Variable(torch.Tensor(np.vstack(obs_n[:, i])), requires_grad=False)
                     for i in range(tmp_env.n)]

        # Get actions from policy network
        action_n = alg_controller.policy(torch_obs, fault_controller, fault_info, training_mode)

        # Reshape actions to match environment structure
        actions = [[ac[i] for ac in action_n] for i in range(args.n_rollout_threads)]

        # Apply action faults if configured
        fault_controller.action_fault_static(action_n, fault_info)

        # Step the environment with generated actions
        new_obs_n, rew_n, done_n, info, terminated, fault_info = env.step(actions)

        # Apply faults if configured
        fault_controller.obs_fault(obs_n, fault_info, alg_controller.obs_fault_modify)
        fault_controller.new_obs_fault(new_obs_n, fault_info, alg_controller.obs_fault_modify)
        fault_controller.action_fault_static(action_n, fault_info)

        # Determine if episodes are over (done or terminated)
        episode_over = done_n[:, 0] | terminated

        # Process step results for each parallel environment
        for env_idx in range(args.n_rollout_threads):
            # Store step information
            step_info = {'rew': rew_n[env_idx], 'fault_info': fault_info[env_idx]}
            ep = episode_buffers[env_idx]

            # Add scenario-specific info to step data
            for k, v in info[env_idx].items():
                step_info[k] = v
            ep['step_infos'][ep['current_idx']] = step_info
            ep['current_idx'] += 1

            # Handle episode completion
            if episode_over[env_idx]:
                # Aggregate episode metrics from step data
                episode_info2metrics(ep['episode_info'], ep['step_infos'][0:ep['current_idx']])
                tmp_env.scenario_info2metrics(ep['episode_info'], ep['step_infos'][0:ep['current_idx']])
                episode_metrics_list.append(ep['episode_info'].copy())

                # Log metrics if not in debug mode
                if not args.debug:
                    wandb.log(ep['episode_info'])

                # Print progress periodically
                if episode_id % 20 == 0:
                    local_print(episode_metrics_list)

                # Prepare for next episode
                episode_id += 1
                ep['current_idx'] = 0
                ep['episode_info'] = {'rew': 0}

        # Collect indices of non-terminated episodes for training data
        sample_indices = ~terminated
        if np.max(sample_indices) > 0:
            # Update replay buffer with transition data
            alg_controller.update_transition(
                obs_n[sample_indices],
                new_obs_n[sample_indices],
                rew_n[sample_indices],
                done_n[sample_indices],
                np.stack(action_n, axis=1)[sample_indices],
                np.array(fault_info)[sample_indices]
            )

        # Update observations for next step
        obs_n = new_obs_n

        # Count valid steps
        step_id += np.sum(sample_indices)

        # Perform algorithm updates at specified frequency
        if (episode_id >= args.learning_start_episode and
                step_id >= args.learning_fre * update_id):
            if update_id == 0:
                step_id = 0  # Reset step counter after first update
            update_id += 1

            # Execute training update
            alg_controller.prep_training(device=args.device)
            alg_controller.update(args, episode_id)

        # Save model at specified intervals
        if (episode_id >= args.start_save_model and
                episode_id >= args.interval_save_model * save_id):
            alg_controller.save_all(args.interval_save_model * save_id, episode_metrics_list)
            save_id += 1

    env.close()


def run_test(alg_controller, fault_controller, episode, args):
    """
    Testing/evaluation loop for trained policies.

    Args:
        alg_controller: Controller for the reinforcement learning algorithm
        fault_controller: Controller for managing fault injection
        episode: Episode number to load model from
        args: Configuration arguments
    """
    # Load trained model
    alg_controller.load_model(episode)
    training_mode = False  # Use deterministic policies during testing

    # Initialize environments and controllers
    cl_controller = make_cl(args.cl, args)
    env = make_parallel_env(args.domain, args.env, args.n_rollout_threads, args, cl_controller)
    tmp_env, kwargs = make_env(args.domain, args.env, args, cl_controller)

    # Initialize testing counters and buffers
    episode_id = 1
    obs_n, fault_info = env.reset()
    if args.display:
        env.render()

    episode_metrics_list = []
    episode_buffers = [{
        'step_infos': [None] * args.per_episode_max_len,
        'current_idx': 0,
        'episode_info': {'rew': 0}
    } for _ in range(args.n_rollout_threads)]

    # Utility for logging attention weights
    att_print = AttentionPrinter(alg_controller, args)

    # Main testing loop
    while episode_id <= args.test_episode:
        # Prepare algorithm for rollouts
        alg_controller.prep_rollouts(args.device)

        # Convert observations to torch Variables
        torch_obs = [Variable(torch.Tensor(np.vstack(obs_n[:, i])), requires_grad=False)
                     for i in range(tmp_env.n)]

        # Get deterministic actions from policy
        action_n = alg_controller.policy(torch_obs, fault_controller, fault_info, training_mode)

        # Reshape actions to match environment structure
        actions = [[ac[i] for ac in action_n] for i in range(args.n_rollout_threads)]

        # Step the environment
        new_obs_n, rew_n, done_n, info, terminated, fault_info = env.step(actions)
        if args.display:
            env.render()

        # Apply observation and action faults
        fault_controller.obs_fault(obs_n, fault_info, alg_controller.obs_fault_modify)
        fault_controller.new_obs_fault(new_obs_n, fault_info, alg_controller.obs_fault_modify)
        fault_controller.action_fault_static(action_n, fault_info)

        # Determine if episodes are over (done or terminated)
        episode_over = done_n[:, 0] | terminated

        # Process step results for each parallel environment
        for env_idx in range(args.n_rollout_threads):
            # Log attention weights if not rendering
            if not args.display:
                att_print.add_critic_att(obs_n[env_idx], actions[env_idx], fault_info[env_idx])
                att_print.add_actor_att(obs_n[env_idx], fault_info[env_idx])

            # Store step information
            step_info = {'rew': rew_n[env_idx], 'fault_info': fault_info[env_idx]}
            ep = episode_buffers[env_idx]

            for k, v in info[env_idx].items():
                step_info[k] = v
            ep['step_infos'][ep['current_idx']] = step_info
            ep['current_idx'] += 1

            # Handle episode completion
            if episode_over[env_idx]:
                # Aggregate episode metrics
                episode_info2metrics(ep['episode_info'], ep['step_infos'][0:ep['current_idx']])
                tmp_env.scenario_info2metrics(ep['episode_info'], ep['step_infos'][0:ep['current_idx']])
                episode_metrics_list.append(ep['episode_info'].copy())

                # Print progress periodically
                if episode_id % 20 == 0:
                    local_print(episode_metrics_list)

                # Prepare for next episode
                episode_id += 1
                ep['current_idx'] = 0
                ep['episode_info'] = {'rew': 0}

        # Update observations for next step
        obs_n = new_obs_n

    # Log final results and attention weights
    if not args.display:
        log_validation(args, episode_metrics_list)
        att_print.print_critic_att()
        att_print.print_actor_att()

    env.close()