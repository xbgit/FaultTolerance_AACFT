import numpy as np
from mpe.core import World, Agent, Landmark, Action
from mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.scenario = 'broken2'
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 5
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i in range(num_good_agents):
            world.agents[num_adversaries + i].action_callback = self.prey_action
        for idx, agent in enumerate(world.agents):
            agent.id = idx
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True if i < num_adversaries else False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.04 if agent.adversary else 0.055
            agent.accel = 3.0 if agent.adversary else 2.0
            agent.max_speed = 1.0 if agent.adversary else 0.7
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        world.t = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85-0.4*(i-4), 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            if agent.id <= 4:
                agent.color = np.array([0.0, 0.0, 0.15 + 0.2 * agent.id])
            agent.collision_times = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1, 1, 1])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.fault = False
            agent.movable = True
            agent.collide = True if agent.adversary else False
            agent.collision_times = 0
            agent.goal = []

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        """Check if two agents collide by comparing their distance with the sum of their sizes."""
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        """Return a list of all non-adversarial (good) agents."""
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """Return a list of all adversarial agents."""
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        """Return agent-specific reward: adversary reward for adversaries, prey reward otherwise."""
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        """(Never used) Reward for prey: penalizes collisions with adversaries and boundary exits; optional distance shaping."""
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        """Calculate reward for adversarial agents. Rewards collisions with good agents;
        penalizes distance from closest uncaptured good agent and exiting the environment boundary."""
        rew = 0

        rew -= np.max((0, abs(agent.state.p_pos[0])-1, abs(agent.state.p_pos[1])-1)) ** 2

        # Adversaries are rewarded for collisions with agents
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        if agent.collision_times == 0:
            min_dis = 2
            for ag in agents:
                tmp_dis = np.sqrt(np.sum(np.square(agent.state.p_pos - ag.state.p_pos)))
                if min_dis > tmp_dis and ag.collision_times == 0:
                    min_dis = tmp_dis
            rew -= min_dis

        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if adv.fault:
                        continue
                    if ag.collision_times < 1:
                        if self.is_collision(ag, adv):
                            ag.movable = False
                            if adv.collision_times == 0:
                                adv.collision_times = 1
                            ag.collision_times = 0.5
                            rew += 10 + 10 * np.sum([ag.collision_times > 0 for ag in agents])

        return rew

    def prey_action(self, agent, world):
        """Define action strategy for good agents (preys). Computes movement direction to avoid adversaries
        and stay within environment boundaries, then generates action probabilities."""
        n = 0
        for i in range(3):
            if not world.agents[i].fault:
                n += 1
        v = np.zeros(2)

        for ag in world.agents:
            if ag.fault or ag == agent:
                continue
            t = agent.state.p_pos - ag.state.p_pos
            tn = np.sum(np.array(t) ** 2)
            t += 0.4 * ((t > 0) - 0.5)

            v += 1 / t / np.array([tn, tn])


        v[0] += 2 * n * (1 / ((agent.state.p_pos[0] + 1) * (agent.state.p_pos[0] + 1 > 0.01) + 0.01) - 1 / (
                    (-agent.state.p_pos[0] + 1) * (-agent.state.p_pos[0] + 1 > 0.01) + 0.01))
        v[1] += 2 * n * (1 / ((agent.state.p_pos[1] + 1) * (agent.state.p_pos[1] + 1 > 0.01) + 0.01) - 1 / (
                    (-agent.state.p_pos[1] + 1) * (-agent.state.p_pos[1] + 1 > 0.01) + 0.01))

        v[0] -= 2000 * ((agent.state.p_pos[0] > 0.95) | (agent.state.p_pos[0] < -0.95)) * agent.state.p_pos[0]
        v[1] -= 2000 * ((agent.state.p_pos[1] > 0.95) | (agent.state.p_pos[1] < -0.95)) * agent.state.p_pos[1]

        model_out = np.array([-1000, v[0] if v[0] > 0 else 0, 0 if v[0] > 0 else -v[0],
                              v[1] if v[1] > 0 else 0, 0 if v[1] > 0 else -v[1]])
        model_out -= np.max(model_out)
        policy = np.exp(model_out) / np.sum(np.exp(model_out))

        action = Action()
        action.u = np.array([policy[1] - policy[2], policy[3] - policy[4]]) * agent.accel
        action.c = np.array([0, 0])
        return action


    def observation(self, agent, world):
        """Generate observation for an agent."""
        if world.t == 0:
            agent.obs = np.ones(26) * 100

        obs = []
        obs.append(agent.state.p_pos)
        obs.append(agent.state.p_vel)

        for ag in world.agents:
            if ag != agent:
                obs.append(ag.state.p_pos - agent.state.p_pos)
                if agent.adversary:
                    if not ag.adversary:
                        obs.append([ag.collision_times])
                    else:
                        obs.append(ag.state.p_vel - agent.state.p_vel)
                        obs.append([int(ag.fault)])

        if agent.adversary:
            for lm in world.landmarks:
                obs.append(lm.state.p_pos - agent.state.p_pos)

        obs = np.concatenate(obs)
        obs_last = agent.obs
        agent.obs = obs

        return agent.obs

    def done_callback(self, agent, world):
        """Determine if the episode is done. Returns True if all good agents have been captured (collision times > 0)."""
        preys = self.good_agents(world)
        for prey in preys:
            if prey.collision_times > 0:
                prey.collision_times = 1
        return all([ag.collision_times > 0 for ag in preys])


    def info_callback(self, agent, world):
        """Return additional info for the agent."""
        preys = self.good_agents(world)
        times_colli = np.average([prey.collision_times for prey in preys])
        return {'t_nocomm': 0, 'times_colli': times_colli, 'times_fix': 0}

    @staticmethod
    def info2metrics(episode_info, step_infos):
        """Convert step-wise info into episode metrics."""
        normal_bool_list = ~np.array(step_infos[-1]['fault_info']['fault_list'])
        num_normal_agents = np.sum(normal_bool_list)
        episode_info['times_colli'] = np.sum(np.array(step_infos[-1]['times_colli'])[normal_bool_list]) / num_normal_agents
        t_nocomm = 0
        for step_info in step_infos:
            t_nocomm += np.sum(step_info['t_nocomm']) / num_normal_agents
        episode_info['t_nocomm'] = t_nocomm
