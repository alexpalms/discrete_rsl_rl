from typing import cast

import torch
from tensordict import TensorDict  # type: ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv


class Agent:
    def __init__(self, env: VecEnv):
        self.env = env
        self.nnodes: int = self.env.n  # pyright:ignore[reportUnkownMemberType]
        self.k: int = self.env.action_space.shape[0]  # pyright:ignore[reportUnkownMemberType]

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        # obs['nodes_infection_probability']: shape (num_sim, n_nodes), torch tensor

        infection_probs = cast(
            torch.Tensor, obs[:, -self.nnodes :]
        )  # (num_sim, n_nodes)
        score: torch.Tensor = cast(
            torch.Tensor,
            obs[:, : self.nnodes],  # (num_sim, n_nodes)
        )
        risk = infection_probs * score

        # Step 1 & 2 combined: top-k values and their indices, sorted in descending order
        # torch.topk already returns sorted results
        _, top_k_indices = torch.topk(risk, self.k, dim=1)

        # Step 3: Convert zero-based indices to one-based indexing
        top_k_indices_one_based = top_k_indices + 1

        return top_k_indices_one_based
