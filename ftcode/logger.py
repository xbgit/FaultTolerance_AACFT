from typing import Dict, List, Any
import numpy as np
import os
import string
import torch
import wandb
from ftcode.configs.get_config import args

def wandb_init(main_fn):
    def wrapper(ex_name, start_episode, *args_, **kwargs):
        if args.test or args.debug:
            main_fn(ex_name, start_episode, *args_, **kwargs)
        else:
            os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY'
            wandb.init(
                project="FaultTolerance",
                config=args,
                name=ex_name
            )
            arti_code = wandb.Artifact('algorithm', type='code')
            arti_code.add_dir('/your_path_to/ftcode')
            wandb.log_artifact(arti_code)
            arti_code = wandb.Artifact('environment', type='code')
            arti_code.add_dir('/your_path_to/mpe')
            wandb.log_artifact(arti_code)
            main_fn(ex_name, start_episode, *args_, **kwargs)
            wandb.finish()
    return wrapper


def log_validation(args, episode_infos: List[Dict[str, Any]]) -> None:
    """Log validation results to file with aggregated metrics.
    
    Args:
        args: Configuration arguments (contains model/env parameters)
        episode_infos: List of dictionaries containing episode-level metrics
    """
    # Extract model name from path (remove numeric suffixes)
    model_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
    os.makedirs('valid', exist_ok=True)
    
    with open(os.path.join('valid', model_name), 'a') as f:
        # Log core experiment parameters
        print(f"Model name: {args.old_model_name}", file=f)
        print(f"Test episodes: {args.test_episode}", file=f)
        print(f"Fault injection time: {args.fault_time}", file=f)

        # Aggregate metrics across all episodes
        test_info = {}
        for episode_info in episode_infos:
            for k, v in episode_info.items():
                test_info[k] = test_info.get(k, 0) + v

        # Calculate success rate based on environment type
        if args.env == 'fix':
            success = sum(
                1 for ei in episode_infos
                if ('times_fix' not in ei or ei['times_fix'] == 1) 
                and ei['times_colli'] == 1
            )
        elif args.env == 'spread':
            success = sum(1 for ei in episode_infos if ei['occupied'] == 2)
        elif args.env == 'broken' or args.env == 'multibroken':
            success = sum(1 for ei in episode_infos if ei['times_colli'] == 1)
        else:
            success = 0
        print(f"Success rate: {success / args.test_episode:.4f}", file=f)

        # Log average values for all metrics
        for k, v in test_info.items():
            print(f"Average {k}: {v / args.test_episode:.4f}", file=f)
        print("-" * 50, file=f)


def local_print(episode_metrics_list: List[Dict[str, Any]]) -> None:
    """Print real-time training metrics (last 200 episodes average).
    
    Args:
        episode_metrics_list: List of dictionaries with step/episode metrics
    """
    metrics_buffer = {}
    metrics_mean = {}

    # Collect metrics from last 200 episodes
    for metrics in episode_metrics_list[-200:]:
        for k, v in metrics.items():
            metrics_buffer.setdefault(k, []).append(v)

    # Calculate mean for non-tensor metrics
    for k, v in metrics_buffer.items():
        if not isinstance(v[0], torch.Tensor):
            metrics_mean[k] = round(np.mean(v), 2)

    # Print formatted metrics (overwrite current line)
    print(f"\rTraining: episode {len(episode_metrics_list)}", end="")
    print(" " * 20, end="")
    for k, v in metrics_mean.items():
        print(f"{k}: {v}  ", end="")


def episode_info2metrics(
    episode_info: Dict[str, Any],
    step_infos: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate step-level metrics into episode-level metrics.
    
    Args:
        episode_info: Base dictionary for episode metrics
        step_infos: List of step-level metrics dictionaries
        
    Returns:
        Updated episode_info with aggregated reward metrics
    """
    # Calculate average reward per normal agent (excluding faulty agents)
    for step_info in step_infos:
        normal_agents = ~np.array(step_info['fault_info']['fault_list'])
        num_normal = np.sum(normal_agents)
        
        if num_normal > 0:
            episode_info['rew'] += np.sum(
                np.array(step_info['rew'])[normal_agents]
            ) / num_normal

    return episode_info


class AttentionPrinter:
    """Utility class for logging and printing attention weights from policy networks."""
    
    def __init__(self, alg_controller, args):
        """Initialize attention logger with policy references.
        
        Args:
            alg_controller: Controller for algorithm/policy networks
            args: Configuration arguments (device/agent count)
        """
        self.alg_controller = alg_controller
        self.args = args
        
        # Buffers for attention weights (pre/post fault injection)
        self.critic_att_pre = []
        self.critic_att_post = []
        self.actor_att_pre = [[] for _ in range(args.num_adversaries)]
        self.actor_att_post = [[] for _ in range(args.num_adversaries)]

    def add_critic_att(self, obs_n: List[np.ndarray], action_n: List[np.ndarray], fault_info: Dict[str, Any]) -> None:
        """Record critic network attention weights for current step.
        
        Args:
            obs_n: List of agent observations
            action_n: List of agent actions
            fault_info: Fault status for current step
        """
        # Convert observations/actions to tensor for forward pass
        obs_tensor = torch.from_numpy(
            np.concatenate(obs_n).reshape(1, -1)
        ).to(self.args.device, torch.float)
        
        action_tensor = torch.from_numpy(
            np.concatenate(action_n).reshape(1, -1)
        ).to(self.args.device, torch.float)

        # Get attention matrix and average over heads
        attn_mat = self.alg_controller.critics_cur[0].attn_mat(obs_tensor, action_tensor)
        attn_mat = attn_mat.detach().cpu().numpy().mean(axis=0)

        # Store pre-fault (no faults) or post-fault (stable fault state) attention
        if not any(fault_info['fault_list']):
            self.critic_att_pre.append(attn_mat)
        elif not fault_info['fault_change'] and np.sum(fault_info['fault_list']) <= 1:
            self.critic_att_post.append(attn_mat)

    def print_critic_att(self) -> None:
        """Print average critic attention weights (pre/post fault)."""
        print(f"Critic attention (pre-fault): {np.mean(self.critic_att_pre, axis=0)}")
        print(f"Critic attention (post-fault): {np.mean(self.critic_att_post, axis=0)}")

    def add_actor_att(self, obs_n: List[np.ndarray], fault_info: Dict[str, Any]) -> None:
        """Record actor network attention weights for current step.
        
        Args:
            obs_n: List of agent observations
            fault_info: Fault status for current step
        """
        for i in range(self.args.num_adversaries):
            # Convert observation to tensor and get attention weights
            obs_tensor = torch.from_numpy(obs_n[i]).to(self.args.device, torch.float)
            attn_mat = self.alg_controller.actors_cur[i].attn_mat(obs_tensor)
            attn_mat = attn_mat.detach().cpu().numpy()[:, 0].mean(axis=0)

            # Store pre/post fault attention weights
            if not any(fault_info['fault_list']):
                self.actor_att_pre[i].append(attn_mat)
            elif not fault_info['fault_change']:
                self.actor_att_post[i].append(attn_mat)

    def print_actor_att(self) -> None:
        """Print average actor attention weights (pre/post fault)."""
        print(f"Actor attention (pre-fault): {np.mean(self.actor_att_pre, axis=1)}")
        print(f"Actor attention (post-fault): {np.mean(self.actor_att_post, axis=1)}")