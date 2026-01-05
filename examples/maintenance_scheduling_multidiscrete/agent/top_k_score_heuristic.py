from typing import cast

import torch
from tensordict import TensorDict  # type: ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv


class Agent:
    def __init__(self, env: VecEnv):
        self.env = env
        self.k = 10
        self.nnodes: int = self.env.n  # type:ignore[reportUnkownMemberType]
        self.action_space_size = cast(int, env.action_space.shape[0])  # type:ignore[reportUnkownMemberType]
        assert self.action_space_size <= self.k, (  # noqa: S101
            f"Action space size ({self.action_space_size}) must be <= top k {self.k}"
        )

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        score = cast(torch.Tensor, obs[:, : self.nnodes])  # (num_sim, n_nodes)

        # Step 1 & 2 combined: top-k values and their indices, sorted in descending order
        # torch.topk already returns sorted results
        _, top_k_indices = torch.topk(score, self.k, dim=1)

        # Step 3: Convert zero-based indices to one-based indexing
        top_k_indices_one_based = top_k_indices + 1
        top_k_indices_one_based = torch.cat(
            [
                top_k_indices_one_based,
                top_k_indices_one_based,
            ],
            axis=-1,
        )

        idx = cast(int, self.env.clock_step) % self.k  # type:ignore[reportUnkownMemberType]

        return top_k_indices_one_based[:, idx : idx + self.action_space_size]
