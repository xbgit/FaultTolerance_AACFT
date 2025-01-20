from ftcode.configs.arguments import parse_args
import os
import yaml

args = parse_args()
env_name = args.env
alg_name = args.alg


alg_file_name = os.path.join('ftcode', 'configs', 'algs', '{}.yaml'.format(alg_name))
if not os.path.exists(alg_file_name):
    print('algorithm {} config file does not exists'.format(alg_name))
else:
    with open(alg_file_name, 'r') as f:
        alg_config_dict = yaml.safe_load(f)
    if alg_config_dict is None:
        print('alg {} config file is emtpy'.format(env_name))
    else:
        for k, v in alg_config_dict.items():
            args.__setattr__(k, v)


env_file_name = os.path.join('ftcode', 'configs', 'envs', '{}.yaml'.format(env_name))
if not os.path.exists(env_file_name):
    print('env {} config file does not exists'.format(env_name))
else:
    with open(env_file_name, 'r') as f:
        env_config_dict = yaml.safe_load(f)
    if env_config_dict is None:
        print('env {} config file is emtpy'.format(env_name))
    else:
        for k, v in env_config_dict.items():
            args.__setattr__(k, v)

