import numpy as np
from gymnasium import spaces

class Agent:
    def __init__(self, env):
        self.env = env
        self.no_actions = np.array([[0] * self.env.action_space.shape[0]] * self.env.num_envs)

    def get_action(self, obs):
        return self.no_actions