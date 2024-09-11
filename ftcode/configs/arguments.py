# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch

import time
import torch
import argparse


time_now = time.strftime('%y%m_%d%H%M')

def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")

    # environment
    parser.add_argument("--domain", type=str, default="mpe", help="the domain which the scenario belongs to")
    parser.add_argument("--env", type=str, default="spread", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--per_episode_max_len", type=int, default=20, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=200000, help="maximum episode length")
    parser.add_argument("--num_adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--homo", type=bool, default=False)

    parser.add_argument("--debug", action="store_true", default=False)

    # core training parameters
    parser.add_argument("--alg", default='acft', help="algorithm name ")
    parser.add_argument("--fault", default='nofault', help="fault name ")
    parser.add_argument("--cl", default='random', help="curriculum name ")
    parser.add_argument("--fault_time", type=int, default=0, help="fault time when cl is fixed")


    parser.add_argument("--device", default='cpu', help="torch device ")
    parser.add_argument("--learning_start_episode", type=int, default=2500, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=2, help="learning frequency")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e5, help="number of data stored in the memory")
    parser.add_argument("--mlp_hidden_size", type=int, default=128, help="number of units in the mlp")

    parser.add_argument("--critic_features_num", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--actor_features_num", type=int, default=64, help="number of units in the mlp")

    parser.add_argument("--head_num", type=int, default=1, help="number of units in the mlp")


    # checkpointing
    parser.add_argument("--interval_save_model", type=int, default=4000, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=200, help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="model", \
            help="directory in which training state and model should be saved")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--old_model_name", type=str, default=None, help="directory in which training state and model are loaded")

    # evaluation
    parser.add_argument("--test_episode", type=int, default=1000)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--att_fig", action="store_true", default=False, help="whether to view attention attribution martix")
    parser.add_argument("--att_critic", action="store_true", default=False, help="whether to cal attention difference pre and post fault")
    parser.add_argument("--att_actor", action="store_true", default=False, help="whether to cal attention difference pre and post fault")


    parser.add_argument("--critic_mode", type=str, default="ac")
    parser.add_argument("--actor_mode", type=str, default="aa")


    return parser.parse_args()
