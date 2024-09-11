from ftcode.fault.registry import FAULTS


@FAULTS.register
class NoFault:
    def __init__(self, env, alg_controller, cl_controller):
        self.n = env.n
        self.agent_list = list(range(self.n))
        self.fault_list = [False] * self.n
        self.agents = env.world.agents
        self.fault_change = False

    def add_fault(self, time_step, obs_n):
        return False

    def obs_fault(self, obs_n):
        pass

    def action_fault(self, action_n):
        pass

    def new_obs_fault(self, new_obs_n):
        pass

    def reset(self):
        self.fault_list = [False] * self.n
        for agent in self.agents:
            agent.fault = False

    def info(self):
        return {'fault_list': self.fault_list, 'fault_change': self.fault_change}
