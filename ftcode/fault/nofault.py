from ftcode.fault.registry import FAULTS


@FAULTS.register
class NoFault:
    """Fault controller for scenarios with no agent faults.

    Maintains a fault-free environment where all agents operate normally.
    Implements the fault controller interface with no-op methods for fault handling.
    """

    def __init__(self, args, env, cl_controller):
        """Initialize the no-fault controller.

        Args:
            args: Configuration parameters
            env: Environment instance
            cl_controller: Curriculum learning controller (unused here)
        """
        self.n = env.n  # Number of agents
        self.agent_list = list(range(self.n))  # List of agent indices
        self.fault_list = [False] * self.n
        self.agents = env.world.agents  # Reference to environment agents
        self.fault_change = False

    def add_fault(self, time_step):
        """No-op method for adding faults (no faults in this mode).

        Args:
            time_step: Current environment time step

        Returns:
            False (no faults added)
        """
        return False

    def reset(self):
        """Reset all agents to non-faulty state.

        Resets fault tracking variables and ensures all agents in the environment
        are marked as non-faulty.
        """
        self.fault_list = [False] * self.n
        for agent in self.agents:
            agent.fault = False  # Reset environment agent fault status

    def info(self):
        """Return current fault information.

        Returns:
            Dictionary containing fault status list and change flag (both indicate no faults)
        """
        return {'fault_list': self.fault_list, 'fault_change': self.fault_change}

    @staticmethod
    def obs_fault(obs_n, fault_info, func_obs_fault_modify):
        """No-op method for modifying observations due to faults.

        Args:
            obs_n: List of observations for each agent
            fault_info: Fault status information (unused)
            func_obs_fault_modify: Function to modify observations (unused)
        """
        pass

    @staticmethod
    def actors_nofault(actors, obs_n, fault_info):
        """No-op method for handling actor behavior with no faults.

        Args:
            actors: List of actor networks
            obs_n: List of observations
            fault_info: Fault status information (unused)
        """
        return

    def action_fault(self, action_n):
        """No-op method for modifying actions due to faults.

        Args:
            action_n: List of actions from each agent (unchanged)
        """
        pass

    @staticmethod
    def new_obs_fault(new_obs_n, fault_info, func_obs_fault_modify):
        """No-op method for modifying new observations due to faults.

        Args:
            new_obs_n: List of new observations after action
            fault_info: Fault status information (unused)
            func_obs_fault_modify: Function to modify observations (unused)
        """
        pass

    @staticmethod
    def action_fault_static(action_n, fault_info):
        """No-op static method for modifying actions due to faults.

        Args:
            action_n: List of actions from each agent (unchanged)
            fault_info: Fault status information (unused)
        """
        pass