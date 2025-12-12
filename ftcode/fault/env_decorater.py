def with_fault_injection(func):
    """Decorator to inject fault handling into environment step method.

    Args:
        func: Original environment step method

    Returns:
        Wrapped step method with fault handling
    """

    def wrapped_step(env, action_n):
        if env.fault_controller:
            env.fault_controller.add_fault(env.time)
            env.fault_controller.action_fault(action_n)
            obs_n, rew_n, done_n, info_n = func(env, action_n)
            fault_info = env.fault_controller.info()
            return obs_n, rew_n, done_n, info_n, fault_info
        else:
            obs_n, rew_n, done_n, info_n = func(env, action_n)
            return obs_n, rew_n, done_n, info_n, {}

    return wrapped_step


def with_fault_reset(func):
    """Decorator to inject fault reset into environment reset method.

    Wraps the environment's reset method to reset the fault controller
    and return initial fault information.

    Args:
        func: Original environment reset method

    Returns:
        Wrapped reset method with fault controller reset
    """

    def wrapped_reset(env):
        obs_n = func(env)
        if env.fault_controller:
            env.fault_controller.reset()
            fault_info = env.fault_controller.info()
            return obs_n, fault_info
        else:
            return obs_n, {}

    return wrapped_reset