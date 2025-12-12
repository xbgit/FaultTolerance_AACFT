import numpy as np
from multiprocessing import Process, Pipe
from ftcode.utils.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            terminated = False
            ob, reward, done, info, fault_info = env.step(data)
            if all(done):
                ob, _ = env.reset()
            if env.time >= env.per_episode_max_len:
                ob, _ = env.reset()
                terminated = True
            remote.send((ob, reward, done, info, terminated, fault_info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, terminated, fault_infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(terminated), fault_infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_n = []
        fault_info_n = []
        for remote in self.remotes:
            result = remote.recv()
            obs_n.append(result[0])
            fault_info_n.append(result[1])
        return np.stack(obs_n), fault_info_n

        # return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def render(self):
        self.envs[0].render()

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos, fault_infos = map(np.array, zip(*results))
        terminated = False
        self.ts += 1
        for (i, done) in enumerate(dones):
            if any(done):
                obs[i], _ = self.envs[i].reset()
                self.ts[i] = 0
            if self.envs[i].time >= self.envs[i].per_episode_max_len:
                obs[i], _ = self.envs[i].reset()
                self.ts[i] = 0
                terminated = True
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos, terminated, fault_infos

    def reset(self):
        obs_n = []
        fault_info_n = []
        for env in self.envs:
            result = env.reset()
            obs_n.append(result[0])
            fault_info_n.append(result[1])
        return np.array(obs_n), fault_info_n

    def close(self):
        return
