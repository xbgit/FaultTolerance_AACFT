from typing import Tuple, Dict, Optional
import mpe.scenarios as scenarios
from mpe.environment import MultiAgentEnv
from ftcode.fault.registry import FAULTS


def make_env(
    domain_name: str,
    scenario_name: str,
    args,
    cl_controller
) -> Tuple[Optional[MultiAgentEnv], Optional[Dict]]:
    """Create multi-agent environment instance with fault injection support.
    
    Args:
        domain_name: Name of the environment domain ('mpe' available)
        scenario_name: Name of the specific scenario ('broken/fix/multi' available)
        args: Configuration arguments
        cl_controller: Curriculum learning controller for fault scheduling
        
    Returns:
        env: Initialized multi-agent environment instance (None if domain invalid)
        kwargs: Dictionary containing observation/action shape info (None if domain invalid)
    """
    if domain_name != 'mpe':
        return None, None

    # Load scenario and initialize world
    scenario = scenarios.load(f"{scenario_name}.py").Scenario()
    world = scenario.make_world()

    # Initialize multi-agent environment with scenario callbacks
    env = MultiAgentEnv(
        world,
        args.per_episode_max_len,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.info_callback,
        scenario.done_callback,
        scenario.pre_step,
        scenario.post_step,
        scenario.info2metrics
    )

    # Attach fault controller to environment
    env.fault_controller = FAULTS[args.fault](args, env, cl_controller)

    # Collect observation/action space dimensions for each agent
    kwargs = {
        'obs_shape_n': [env.observation_space[i].shape[0] for i in range(env.n)],
        'action_shape_n': [env.action_space[i].n for i in range(env.n)],
    }

    return env, kwargs

