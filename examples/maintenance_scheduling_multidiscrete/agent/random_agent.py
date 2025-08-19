import numpy as np
from gymnasium import spaces

class Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return np.array([self.env.action_space.sample()] * self.env.num_envs)