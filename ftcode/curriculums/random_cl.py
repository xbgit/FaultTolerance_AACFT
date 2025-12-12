from ftcode.curriculums.cl_controller import Curriculum
import numpy as np

class RandomCurriculum(Curriculum):
    def __init__(self, args):
        super().__init__(args)
        self.fault_time_range = args.fault_time_range

    def get_fault_time(self, current_episode):
        return np.random.randint(self.fault_time_range[0], self.fault_time_range[1])
