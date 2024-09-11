from ftcode.curriculums.cl_controller import Curriculum
import numpy as np

class RandomCurriculum(Curriculum):
    def __init__(self, args):
        super().__init__(args)

    def get_fault_time(self, current_episode):
        return np.random.randint(0, self.per_episode_max_len)
