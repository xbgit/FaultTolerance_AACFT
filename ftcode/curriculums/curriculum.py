import random
import numpy as np
from ftcode.curriculums.random_cl import RandomCurriculum
from ftcode.curriculums.fixed_cl import FixedCurriculum
from ftcode.curriculums.incremental_cl import IncrementalCurriculum


def make_cl(cl_name, args):
    if cl_name == 'random':
        return RandomCurriculum(args)
    if cl_name == 'incre':
        return IncrementalCurriculum(args)
    if cl_name == 'fixed':
        return FixedCurriculum(args)
