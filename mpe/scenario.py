import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    # create initial conditions of the world
    def info_callback(self, agent, world):
        raise NotImplementedError()

    def done_callback(self, agent, world):
        raise NotImplementedError()

    def pre_step(self, world):
        pass

    def post_step(self, world):
        pass
