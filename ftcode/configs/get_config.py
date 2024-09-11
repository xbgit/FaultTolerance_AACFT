from ftcode.configs.arguments import parse_args
import os
import yaml

args = parse_args()
env_name = args.env
alg_name = args.alg

with open(os.path.join('ftcode', 'configs', 'algs', '{}.yaml'.format(alg_name)), 'r') as f:
    alg_config_dict = yaml.safe_load(f)

if alg_config_dict is not None:
    for k, v in alg_config_dict.items():
        args.k = v

# args.old_model_name = 'broken_acft_s0_2408_131424/33000'

# with open(os.path.join('configs', 'envs', '{}.yaml'.format(env_name)), 'r') as f:
#     env_config_dict = yaml.safe_load(f)
#
# for k, v in env_config_dict.items():
#     args.k = v

