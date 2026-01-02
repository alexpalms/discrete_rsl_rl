"""Deep RL SB3 Agent."""

import os
from typing import cast

import numpy as np
import torch
from stable_baselines3 import PPO
from tensordict import TensorDict  # type: ignore[reportMissingStubs]


class Agent:
    """Deep RL SB3 Agent."""

    def __init__(
        self,
    ) -> None:
        model_path = os.path.join(os.path.dirname(__file__), "model_sb3.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.agent = PPO.load(model_path, device="cuda")  # pyright: ignore[reportUnknownMemberType]

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        """
        Get the action from the policy.

        Parameters
        ----------
        obs : TensorDict
            The observation.
        """
        obs_numpy = cast(dict[str, np.ndarray], obs.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]
        prediction, _ = self.agent.predict(obs_numpy, deterministic=False)
        prediction_tensor = torch.tensor(prediction, dtype=torch.long, device="cuda")
        return prediction_tensor
