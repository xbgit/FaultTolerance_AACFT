from ftcode.configs.get_config import args
from ftcode.domain import make_env
from ftcode.algorithms.algorithm import make_alg
from runner import run_train, run_test
import time
import string
from ftcode.utils.set_seed import seed_all
import multiprocessing as mp
from ftcode.curriculums.curriculum import make_cl
from ftcode.fault.registry import FAULTS
from ftcode.logger import wandb_init

@wandb_init
def main(ex_name, start_episode):
    seed_all(args.seed)
    run = run_train if not args.test else run_test
    cl_controller = make_cl(args.cl, args)
    tmp_env, kwargs = make_env(args.domain, args.env, args, cl_controller)
    alg_controller = make_alg(args.alg, args, kwargs, ex_name)
    fault_controller = FAULTS[args.fault](args, tmp_env, cl_controller)
    run(alg_controller, fault_controller, start_episode, args)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    if args.old_model_name is None:
        start_episode = 0
        time_now = time.strftime('%y%m_%d%H%M')
        ex_name = '{}_{}_s{}_{}_{}'.format(args.env, args.alg, args.seed, time_now, args.fault)
    else:
        ex_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
        start_episode = int(args.old_model_name.split('/')[-1])

    main(ex_name, start_episode)

