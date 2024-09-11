from ftcode.configs.get_config import args
from ftcode.domain import make_env
from ftcode.algorithms.algorithm import make_alg
from ftcode.curriculums.curriculum import make_cl
from ftcode.fault.registry import FAULTS
from ftcode.runner import run_test
import string

ex_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
episode = int(args.old_model_name.split('/')[-1])

# ex_name = '{}_{}_s{}_{}'.format(args.env, args.alg, args.seed, time_now)
cl_controller = make_cl(args.cl, args)
env, kwargs = make_env(args.domain, args.env)

alg_controller = make_alg(args.alg, args, kwargs, ex_name)
fault_controller = FAULTS[args.fault](env, alg_controller, cl_controller)

run_test(env, alg_controller, fault_controller, episode, args)

