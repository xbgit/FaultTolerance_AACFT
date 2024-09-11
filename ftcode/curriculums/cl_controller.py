
class Curriculum:
    def __init__(self, args):
        self.args = args
        self.per_episode_max_len = args.per_episode_max_len
        self.max_episode = args.max_episode
