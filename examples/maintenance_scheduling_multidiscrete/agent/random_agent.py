import numpy as np
import torch

class Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return torch.randint(0, self.env.action_space.nvec[0], (self.env.num_envs, self.env.action_space.shape[0]), device=self.env.device)