from ftcode.curriculums.cl_controller import Curriculum
import numpy as np


class IncrementalCurriculum(Curriculum):
    def __init__(self, args):
        super().__init__(args)

    def get_fault_time(self, current_episode):
        cp1, cp2 = int(self.max_episode / 4), int(self.max_episode * 2 / 4)
        if current_episode < cp1:
            fault_time = self.per_episode_max_len
        elif current_episode < cp2:
            fault_time = self.per_episode_max_len * (cp2 - current_episode) / (cp2 - cp1)
            fault_time *= (1 + (np.random.rand() - 0.5) * 0.2)
        else:
            fault_time = np.random.randint(0, self.per_episode_max_len)

        return fault_time
