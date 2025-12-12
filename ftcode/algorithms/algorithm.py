from ftcode.algorithms.alg_controller import AlgController
from ftcode.algorithms.maddpg import MADDPGLearner
from ftcode.algorithms.aacft import AACFTLearner

def make_alg(alg_name, args, kwargs, ex_name):
    if alg_name == 'Random':
        return AlgController(args, kwargs, ex_name)
    if alg_name == 'maddpg':
        return MADDPGLearner(args, kwargs, ex_name)
    if alg_name == 'acft':
        return AACFTLearner(args, kwargs, ex_name, actor_attention=False, critic_attention=True)
    if alg_name == 'aaft':
        return AACFTLearner(args, kwargs, ex_name, actor_attention=True, critic_attention=False)
    if alg_name == 'aacft':
        return AACFTLearner(args, kwargs, ex_name, actor_attention=True, critic_attention=True)
