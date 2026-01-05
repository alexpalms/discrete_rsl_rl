import torch
from tensordict import TensorDict  # pyright:ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv  # pyright:ignore[reportMissingTypeStubs]


class Agent:
    def __init__(self, env: VecEnv):
        self.env = env

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        return torch.randint(
            0,
            self.env.action_space.nvec[0],  # pyright:ignore[reportUnknownMemberType,reportUnknownArgumentType,reportAttributeAccessIssue]
            (self.env.num_envs, self.env.action_space.shape[0]),  # pyright:ignore[reportUnknownMemberType,reportUnknownArgumentType,reportAttributeAccessIssue]
            device=self.env.device,
        )
