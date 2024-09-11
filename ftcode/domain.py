from mpe.environment import MultiAgentEnv
import mpe.scenarios as scenarios

def make_env(domain_name, scenario_name):
    # if domain_name == 'mpe':
    #     if scenario_name == 'simple_tag':
    #         env = simple_tag_v3.env(render_mode='human')
    #     env.reset()
    #
    #     kwargs = {'obs_shape_n': [env.observation_space(env.agents[i]).shape[0] for i in range(env.num_agents)],
    #               'action_shape_n': [env.action_space(env.agents[i]).n for i in range(env.num_agents)],
    #               'learnable_n': 3,
    #               }
    # return env, kwargs
    if domain_name == 'mpe':
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.info_callback,
                                scenario.done_callback, scenario.pre_step, scenario.post_step, scenario.info2metrics)
        kwargs = {'obs_shape_n': [env.observation_space[i].shape[0] for i in range(env.n)],
                  'action_shape_n': [env.action_space[i].n for i in range(env.n)],
                  }
    else:
        env = None
        kwargs = None

    return env, kwargs

if __name__ == '__main__':
    env = make_env('mpe', 'simple_tag')
    print(1)
