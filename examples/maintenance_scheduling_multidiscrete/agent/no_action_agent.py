import numpy as np
import torch

class Agent:
    def __init__(self, env):
        self.env = env
        self.no_actions = torch.zeros(self.env.num_envs, self.env.action_space.shape[0], device=self.env.device)

    def get_action(self, obs):
        return self.no_actions