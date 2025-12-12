from ftcode.curriculums.cl_controller import Curriculum
import numpy as np


class FixedCurriculum(Curriculum):
    def __init__(self, args):
        super().__init__(args)

    def get_fault_time(self, current_episode=None):
        fault_time = self.args.fault_time
        return fault_time
