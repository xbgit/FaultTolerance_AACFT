import numpy as np
from mpe.core import World, Agent, Landmark, Action
from mpe.scenario import BaseScenario
from scipy.spatial.distance import pdist
import torch

class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.world = World()

    def make_world(self):
        world = self.world
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i in range(num_good_agents):
            world.agents[num_adversaries + i].action_callback = self.prey_action
        for idx, agent in enumerate(world.agents):
            agent.id = idx
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.id == 0 else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        # make initial conditions
        world.commState = np.zeros((num_adversaries, num_adversaries))
        world.obsState = np.zeros(num_adversaries)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.t = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            if agent.id == 0:
                agent.color = np.array([0.0, 0.0, 0.35])
            if agent.id == 1:
                agent.color = np.array([0.0, 0.0, 0.65])
            if agent.id == 2:
                agent.color = np.array([0.0, 0.0, 0.95])
            # random properties for landmarks
            agent.collision_times = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1, 1, 1])
        # set random initial states
        world.commState = np.zeros((3,3))
        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.fault = False
            agent.movable = True
            agent.collide = True
            agent.goal = []
        while True:
            world.agents[3].state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)

            p = []
            p.append(np.random.uniform(-1, +1, world.dim_p))
            p.append(np.array([np.random.uniform(max(-1, p[0][0] - 0.35), min(1, p[0][0] + 0.35)), \
                np.random.uniform(max(-1, p[0][1] - 0.35), min(1, p[0][1] + 0.35))]))
            p.append(np.array([np.random.uniform(max(-1, p[1][0] - 0.35), min(1, p[1][0] + 0.35)), \
                np.random.uniform(max(-1, p[1][1] - 0.35), min(1, p[1][1] + 0.35))]))

            dis_limit = 0.25 ** 2
            if (np.square(p[0][0] - p[1][0]) + np.square(p[0][1] - p[1][1])) < dis_limit or \
                (np.square(p[1][0] - p[2][0]) + np.square(p[1][1] - p[2][1])) < dis_limit or \
                (np.square(p[0][0] - p[2][0]) + np.square(p[0][1] - p[2][1])) < dis_limit:
                continue

            id1 = np.random.randint(0, 3)
            id2 = (id1 + np.random.randint(1, 3)) % 3
            id3 = 3 - id1 - id2
            world.agents[0].state.p_pos = p[id1]
            world.agents[1].state.p_pos = p[id2]
            world.agents[2].state.p_pos = p[id3]

            world.commState = np.zeros((3, 3))
            self.get_comm_world(world)
            break
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                
                landmark.state.p_vel = np.zeros(world.dim_p)
        # world.landmarks[0].state.p_pos = np.array([-0.5, 0.5])
        # world.landmarks[1].state.p_pos = np.array([0.5, -0.5])

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
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
        rew = 0
        if agent.id == 0:
            if agent.comm[0] == 0 and not world.agents[1].fault:
                rew -= 2
            if agent.comm[3] == 0 and not world.agents[2].fault:
                rew -= 2
        if agent.id == 1:
            if agent.comm[0] == 0 and not world.agents[0].fault:
                rew -= 2
            if agent.comm[3] == 0 and not world.agents[2].fault:
                rew -= 2
        if agent.id == 2:
            if agent.comm[0] == 0 and not world.agents[0].fault:
                rew -= 2
            if agent.comm[3] == 0 and not world.agents[1].fault:
                rew -= 2

        rew -= np.max((0, abs(agent.state.p_pos[0])-1, abs(agent.state.p_pos[1])-1)) ** 2
        # rew = 0
        rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[3].state.p_pos)))
        # if agent.id != 0:
        #     return rew
        # Adversaries are rewarded for collisions with agents
        agent.collision_times = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 50
                        agent.collision_times = 1

        for adv in adversaries:
            if agent != adv and not adv.fault and self.is_collision(agent, adv):
                rew -= 1

        return rew

    def get_comm_world(self, world):
        adversaries = self.adversaries(world)
        for idx1, agent1 in enumerate(adversaries):
            for idx2, agent2 in enumerate(adversaries):
                if idx1 == idx2: continue
                delta_pos = agent1.state.p_pos - agent2.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                if world.commState[idx1, idx2] >= 0 and agent2.fault:
                    if world.commState[idx1, idx2] == 0:
                        world.commState[idx1, idx2] = 1
                    world.commState[idx1, idx2] += world.dt

                if world.commState[idx1, idx2] <= 0:
                    if dist > world.commRange:
                        if world.commState[idx1, idx2] == 0:
                            world.commState[idx1, idx2] = -1
                        world.commState[idx1, idx2] -= world.dt
                    else:
                        world.commState[idx1, idx2] = 0

    def get_prey(self, world):
        for i in range(3):
            delta_pos = world.agents[i].state.p_pos - world.agents[3].state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            world.obsState[i] = (0 if dist > world.obsRange else 1)
        world.obsState[0] = 1

    def get_comm(self, agent, world):
        # return np.array([1,0,0,1,0,0])
        ls = [0,1,2]
        ls.pop(agent.id)
        for i, id in enumerate(ls):
            if world.commState[agent.id, id] < 0 and world.commState[agent.id, 3-id-agent.id] == 0:
                if world.commState[3-id-agent.id, id] == 0:
                    world.commState[agent.id, id] = 0
                elif world.commState[3-id-agent.id, id] < 0 and world.commState[agent.id, id] < world.commState[3-id-agent.id, id]:
                    world.commState[agent.id, id] = world.commState[3-id-agent.id, id]
                    # agent.obs[4+i*7:8+i*7] = world.agents[3-id-agent.id].obs[4+i*7:8+i*7]
                    agent.obs[4+i*7:8+i*7] = (world.agents[3-id-agent.id].obs[4+i*7:8+i*7] + np.array([world.agents[3-id-agent.id].state.p_pos, world.agents[3-id-agent.id].state.p_vel]).reshape(-1)
                                              - np.array([agent.state.p_pos, agent.state.p_vel]).reshape(-1))
                elif world.commState[3-id-agent.id, id] > 0:
                    world.commState[agent.id, id] = world.commState[3-id-agent.id, id]
                    # agent.obs[4+i*7:8+i*7] = world.agents[3-id-agent.id].obs[4+i*7:8+i*7]
                    agent.obs[4+i*7:8+i*7] = (world.agents[3-id-agent.id].obs[4+i*7:8+i*7] + np.array([world.agents[3-id-agent.id].state.p_pos, world.agents[3-id-agent.id].state.p_vel]).reshape(-1)
                                              - np.array([agent.state.p_pos, agent.state.p_vel]).reshape(-1))
        comm = np.zeros(6)
        for i in range(2):
            if world.commState[agent.id, ls[i]] == 0:
                comm[3*i] = 1
            elif world.commState[agent.id, ls[i]] < 0:
                comm[3*i + 1] = -world.commState[agent.id, ls[i]]
            else:
                comm[3*i + 2] = world.commState[agent.id, ls[i]]
        return comm

    def get_goal(self, agent, world):
        if agent.adversary:
            return []
        if not world.agents[2].fault and not world.agents[1].fault:
            agent.goal = [-10, -10]
            return agent.goal
        if agent.goal != [-10, -10]:
            return agent.goal

        p1 = world.agents[0].state.p_pos
        if world.agents[1].fault:
            p2 = world.agents[2].state.p_pos
        else:
            p2 = world.agents[1].state.p_pos

        p = agent.state.p_pos
        danger = np.linalg.norm(p - p1) < 0.3 or np.linalg.norm(p - p2) < 0.3
        if p1[0] > 0 and p2[0] > 0 and p1[1] > 0 and p2[1] > 0 and danger:
            r0 = 1 * ((1-p1[1]) / np.linalg.norm(p - p1) + (1-p2[1]) / np.linalg.norm(p - p2))
            r1 = 1 * ((1-p1[0]) / np.linalg.norm(p - p1) + (1-p2[0]) / np.linalg.norm(p - p2))
            r2 = np.linalg.norm(p - [0.9, 0.9])
            g0, g1, g2 = [-0.9, 0.9], [0.9, -0.9], [0.9, 0.9]
        elif p1[0] > 0 and p2[0] > 0 and p1[1] < 0 and p2[1] < 0 and danger:
            r0 = 1 * ((1 + p1[1]) / np.linalg.norm(p - p1) + (1 + p2[1]) / np.linalg.norm(p - p2))
            r1 = 1 * ((1 - p1[0]) / np.linalg.norm(p - p1) + (1 - p2[0]) / np.linalg.norm(p - p2))
            r2 = np.linalg.norm(p - [0.9, -0.9])
            g0, g1, g2 = [-0.9, -0.9], [0.9, 0.9], [0.9, -0.9]
        elif p1[0] < 0 and p2[0] < 0 and p1[1] < 0 and p2[1] < 0 and danger:
            r0 = 1 * ((1 + p1[1]) / np.linalg.norm(p - p1) + (1 + p2[1]) / np.linalg.norm(p - p2))
            r1 = 1 * ((1 + p1[0]) / np.linalg.norm(p - p1) + (1 + p2[0]) / np.linalg.norm(p - p2))
            r2 = np.linalg.norm(p - [-0.9, -0.9])
            g0, g1, g2 = [-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9]
        elif p1[0] < 0 and p2[0] < 0 and p1[1] > 0 and p2[1] > 0 and danger:
            r0 = 1 * ((1 - p1[1]) / np.linalg.norm(p - p1) + (1 - p2[1]) / np.linalg.norm(p - p2))
            r1 = 1 * ((1 + p1[0]) / np.linalg.norm(p - p1) + (1 + p2[0]) / np.linalg.norm(p - p2))
            r2 = np.linalg.norm(p - [-0.9, -0.9])
            g0, g1, g2 = [0.9, 0.9], [-0.9, -0.9], [-0.9, 0.9]
        else:
            return [-10, -10]

        r3 = 1 - abs(1 - pdist([p - p1, p - p2], 'cosine'))
        r = np.random.rand() * (r0 + r1 + r2 + r3)
        if r < r0:
            agent.goal = g0
        elif r < r0 + r1:
            agent.goal = g1
        elif r < r0 + r1 + r2:
            agent.goal = g2
        else:
            agent.goal = list(p1 + p2 - p)

        return agent.goal

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        if world.t == 0:
            agent.obs = np.ones(26) * 100
        if agent.id == 0:
            self.get_comm_world(world)
            self.get_prey(world)

        # communication of all other agents
        if agent.adversary:
            comm = self.get_comm(agent, world)
            agent.comm = comm

        obs = []
        obs.append(agent.state.p_pos)
        obs.append(agent.state.p_vel)

        idx = 0
        for ag in world.agents:
            if ag != agent:
                obs.append(ag.state.p_pos - agent.state.p_pos)
                obs.append(ag.state.p_vel - agent.state.p_vel)
                if ag.adversary and agent.adversary:
                    obs.append(comm[idx*3: (idx+1)*3])
                    idx += 1

        goal = self.get_goal(agent, world)
        # goal = [-10, -10] if not agent.adversary else []
        obs.append(goal)

        if agent.adversary:
            for lm in world.landmarks:
                obs.append(lm.state.p_pos - agent.state.p_pos)

        obs = np.concatenate(obs)
        obs_last = agent.obs
        agent.obs = obs

        # 未通信
        if agent.adversary:
            for i in range(2):
                if not comm[i*3] == 1:
                    if (obs[9+i*7] > 0 and obs_last[9+i*7] > 0) or (obs[10+i*7] > 0 and obs_last[10+i*7] > 0):
                        agent.obs[4+i*7:6+i*7] = obs_last[4+i*7:6+i*7] + obs_last[0:2] - obs[0:2]

        # 未感知
        if agent.adversary:
            if agent.id == 0:
                tmp_comm = np.array([1, comm[0], comm[3]])
            elif agent.id == 1:
                tmp_comm = np.array([comm[0], 1, comm[3]])
            elif agent.id == 2:
                tmp_comm = np.array([comm[0], comm[3], 1])

            if np.dot(tmp_comm, world.obsState) == 0:
                agent.obs[18:22] = obs_last[18:22]
                agent.obs_prey = False
            else:
                agent.obs_prey = True
        return agent.obs

    def done_callback(self,agent,world):
        # for other in world.agents:
        #     if not other.adversary == agent.adversary:
        #         if self.is_collision(agent,other):
        #             return True
        for a in world.agents:
            for b in world.agents:
                if a.adversary != b.adversary:
                    if self.is_collision(a, b):
                        return True
        return False

    def info_callback(self,agent,world):

        if agent.id == 3:
            t_nocomm = 0
            times_colli = 0
        elif world.agents[2].fault:
            t_nocomm = np.sum(np.array(agent.comm[0]) == 0)
            times_colli = agent.collision_times
        elif world.agents[1].fault:
            t_nocomm = np.sum(np.array(agent.comm[0 if agent.id == 2 else 3]) == 0)
            times_colli = agent.collision_times
        else:
            t_nocomm = np.sum(np.array(agent.comm[0]) == 0) + np.sum(np.array(agent.comm[3]) == 0)
            times_colli = agent.collision_times
        return {'t_nocomm': t_nocomm / 10, 'times_colli': times_colli, 'times_fix': 0}

    def prey_action(self, agent, world):
        action = Action()
        action.u = np.array([0, 0])
        action.c = np.array([0, 0])
        return action

        # device = 'cpu'
        # n = 3  # if agent.obs.shape[1] == 18 else 2
        # v = torch.zeros((agent.obs.shape[0], 2)).to(device)
        # agent.obs_bool = torch.zeros(agent.obs.shape[0], dtype=bool).unsqueeze(1).to(device)
        # for i in range(n):
        #     t = -agent.obs[:, 4 + 4 * i:6 + 4 * i]
        #     tn = torch.norm(t, dim=1).unsqueeze(1) ** 2
        #     agent.obs_bool |= (tn < 0.5)
        #     t[t[:, 0] > 0, 0] += 0.2
        #     t[t[:, 1] > 0, 1] += 0.2
        #     t[t[:, 0] < 0, 0] -= 0.2
        #     t[t[:, 1] < 0, 1] -= 0.2
        #
        #     v += 1 / t / torch.cat((tn, tn), dim=1)
        #
        # if agent.obs.shape[0] == 1:
        #     agent.obs_bool = torch.squeeze(agent.obs_bool)
        #
        # v[agent.obs[:, 16] > -5] += 100 * (agent.obs[agent.obs[:, 16] > -5, -2:] - agent.obs[agent.obs[:, 16] > -5, 2:4])
        #
        # v[:, 0] += n * (1 / ((agent.obs[:, 0] + 1) * (agent.obs[:, 0] + 1 > 0.01) + 0.01) - 1 / (
        #             (-agent.obs[:, 0] + 1) * (-agent.obs[:, 0] + 1 > 0.01) + 0.01))
        # v[:, 1] += n * (1 / ((agent.obs[:, 1] + 1) * (agent.obs[:, 1] + 1 > 0.01) + 0.01) - 1 / (
        #             (-agent.obs[:, 1] + 1) * (-agent.obs[:, 1] + 1 > 0.01) + 0.01))
        #
        # v[:, 0] -= 1000 * ((agent.obs[:, 0] > 1) | (agent.obs[:, 0] < -1)) * agent.obs[:, 0]
        # v[:, 1] -= 1000 * ((agent.obs[:, 1] > 1) | (agent.obs[:, 1] < -1)) * agent.obs[:, 1]
        #
        # model_out = torch.cat(
        #     (-1000 * torch.ones_like(v[:, 0]), v[:, 0], torch.zeros_like(v[:, 0]), v[:, 1], torch.zeros_like(v[:, 0])),
        #     dim=0).reshape(5, -1).T
        # model_out[model_out[:, 1] < 0, 1:3] += -model_out[model_out[:, 1] < 0, 1:2]
        # model_out[model_out[:, 3] < 0, 3:5] += -model_out[model_out[:, 3] < 0, 3:4]
        # model_out = torch.squeeze(model_out)
        # policy = F.softmax(model_out, dim=-1)


