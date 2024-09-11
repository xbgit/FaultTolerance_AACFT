import wandb
import numpy as np
import os
import string
from utils.timer import timer


def log_validation(args, episode_infos):
    model_name = args.old_model_name.rstrip('/').rstrip(string.digits).rstrip('/')
    f = open(os.path.join('valid', model_name), 'a')
    print('model_name: ', args.old_model_name, file=f)
    print('test_episode: ', args.test_episode, file=f)
    print('fault_time: ', args.fault_time, file=f)
    test_info = {}
    for episode_info in episode_infos:
        for k, v in episode_info.items():
            if k not in test_info.keys():
                test_info[k] = 0
            test_info[k] += v
    for k, v in test_info.items():
        print('{}: {}'.format(k, v / args.test_episode), file=f)


def episode_info2metrics(episode_info, step_infos):
    rew_total, rew_nofault, rew_fault = 0.0, 0.0, 0.0
    t_nocomm, t_nocomm_nofault, t_nocomm_fault = 0.0, 0.0, 0.0
    times_colli = 0

    for step_info in step_infos:
        normal_bool_list = ~np.array(step_info['fault_info']['fault_list'])
        num_normal_agents = np.sum(normal_bool_list)

        episode_info['rew'] += np.sum(np.array(step_info['rew'])[normal_bool_list]) / num_normal_agents

        # t_nocomm += np.sum(np.array(step_info['t_nocomm'])[normal_bool_list]) / num_normal_agents
        # times_colli += np.sum(np.array(step_info['times_colli'])[normal_bool_list]) / num_normal_agents

    # episode_metrics = {'rew': rew_total,
    #                    't_nocomm': t_nocomm,
    #                    'times_colli': times_colli}

    return episode_info


def run_episode(episode_id, env, alg_controller, fault_controller, args, training_mode=True):
    obs_n = env.reset()
    fault_controller.reset()
    done = False
    time_step = 0
    step_infos = []
    while not done:
        fault_controller.add_fault(time_step, obs_n)
        fault_controller.obs_fault(obs_n)

        action_n = alg_controller.policy(obs_n, training_mode)

        fault_controller.action_fault(action_n)

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        fault_controller.new_obs_fault(new_obs_n)
        fault_info = fault_controller.info()

        # if any(fault_info['fault_list']):
        #     print('fault')

        alg_controller.update_transition(obs_n, new_obs_n, rew_n, done_n, action_n, fault_info)
        done = all(done_n) or (time_step >= args.per_episode_max_len - 1)
        policy_updated = False

        obs_n = new_obs_n
        time_step += 1

        step_info = {'episode_id': episode_id, 'rew': rew_n, 'fault_info': fault_info}
        for k, v in info_n.items():
            step_info[k] = v
        step_infos.append(step_info)

        if args.display:
            env.render()
            print(rew_n)

    if training_mode and episode_id >= args.learning_start_episode and episode_id % args.learning_fre == 0:
        alg_controller.update(args, episode_id)

    episode_info = {'rew': 0}
    episode_info2metrics(episode_info, step_infos)
    env.scenario_info2metrics(episode_info, step_infos)
    if episode_id % 100 == 0:
        alg_controller.alg_info2metrics(episode_info)
    
    return episode_info


def run(env, controller, fault_controller, start_episode, args):
    if start_episode > 0:
        controller.load_all(start_episode)
    for episode_id in range(start_episode + 1, args.max_episode + 1):
        episode_metrics = run_episode(episode_id, env, controller, fault_controller, args, training_mode=True)
        if not args.debug:
            wandb.log(episode_metrics)

        if episode_id % args.interval_save_model == 0:
            controller.save_all(episode_id)
            print(episode_id)


def run_test(env, controller, fault_controller, episode, args):

    controller.load_model(episode)
    episode_infos = []
    for episode_id in range(args.test_episode):
        episode_info = run_episode(episode_id, env, controller, fault_controller, args, training_mode=False)
        episode_infos.append(episode_info)
    if not args.display:
        log_validation(args, episode_infos)
