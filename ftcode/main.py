import wandb
from ftcode.configs.get_config import args
from ftcode.domain import make_env
from ftcode.algorithms.algorithm import make_alg
from ftcode.fault.registry import FAULTS
from ftcode.curriculums.curriculum import make_cl
from runner import run
import time
import string
import random
import numpy as np
import torch


# os.environ["WANDB_API_KEY"] = '88ff66bce8b622fb5cd40bbe8c7f958d5b572e47'
# os.environ["WANDB_MODE"] = "offline"

if args.old_model_name is None:
    start_episode = 0
    time_now = time.strftime('%y%m_%d%H%M')
    ex_name = '{}_{}_s{}_{}'.format(args.env, args.alg, args.seed, time_now)
else:
    ex_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
    start_episode = int(args.old_model_name.split('/')[-1])

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if not args.debug:
    wandb.init(
        project="FaultTolerance",
        config=args,
        name=ex_name
    )
    arti_code = wandb.Artifact('algorithm', type='code')
    arti_code.add_dir('/home/syc/Workspace/FaultTolerance/ftcode')
    wandb.log_artifact(arti_code)
    arti_code = wandb.Artifact('environment', type='code')
    arti_code.add_dir('/home/syc/Workspace/FaultTolerance/mpe')
    wandb.log_artifact(arti_code)

cl_controller = make_cl(args.cl, args)

env, kwargs = make_env(args.domain, args.env)

alg_controller = make_alg(args.alg, args, kwargs, ex_name)
fault_controller = FAULTS[args.fault](env, alg_controller, cl_controller)

run(env, alg_controller, fault_controller, start_episode, args)

wandb.finish()
