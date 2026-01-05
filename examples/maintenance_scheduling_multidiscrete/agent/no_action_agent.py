import torch
from tensordict import TensorDict  # pyright:ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv  # pyright:ignore[reportMissingTypeStubs]


class Agent:
    def __init__(self, env: VecEnv):
        self.env = env
        self.no_actions = torch.zeros(
            self.env.num_envs,
            self.env.action_space.shape[0],  # pyright:ignore[reportUnknownArgumentType,reportUnknownMemberType,reportAttributeAccessIssue]
            device=self.env.device,
        )

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        return self.no_actions
