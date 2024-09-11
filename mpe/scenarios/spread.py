import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario
from ftcode.utils.timer import timer

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 2
        # world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.075
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 0.35 + 0.3 * i])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.0, 0.35 + 0.3 * i, 0.0])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.occupied = [False] * len(world.landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # rew_list = [-5, -3, -1, 40]
        for i, landmark in enumerate(world.landmarks):
            if not self.occupied[i]:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.good_agents]
                rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        if all(self.occupied):
            rew += 40

        return rew

    def observation(self, agent, world):
        others_state = []
        for other in world.agents:
            if other is agent: continue
            others_state.append(other.state.p_pos - agent.state.p_pos)
            others_state.append(other.state.p_vel - agent.state.p_vel)
            others_state.append([int(other.fault)])

        # get positions of all entities in this agent's reference frame
        landmark_pos = []
        for entity in world.landmarks:  # world.entities:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents

        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + others_state + landmark_pos)

    def pre_step(self, world):
        pass

    def post_step(self, world):
        for i, landmark in enumerate(world.landmarks):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
            if min(dists) < 0.1:
                self.occupied[i] = True

    def done_callback(self, agent, world):
        return all(self.occupied)

    def info_callback(self, agent, world):
        return {'occupied': np.sum(self.occupied)}

    @staticmethod
    def info2metrics(episode_info, step_infos):
        normal_bool_list = ~np.array(step_infos[-1]['fault_info']['fault_list'])
        num_normal_agents = np.sum(normal_bool_list)
        episode_info['occupied'] = np.sum(np.array(step_infos[-1]['occupied'])[normal_bool_list]) / num_normal_agents
