"""Deep RL RSL Agent."""

import os
from typing import cast

import torch
import yaml
from tensordict import TensorDict  # pyright:ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv  # pyright:ignore[reportMissingTypeStubs]
from rsl_rl.runners import OnPolicyRunner  # pyright:ignore[reportMissingTypeStubs]


class Agent:
    """
    Deep RL RSL Agent.

    Parameters
    ----------
    env : Environment
        The environment to run the agent in.
    model_path : str
        The path to the model file.
    config_path : str
        The path to the configuration file.
    device : str
        The device to run the agent on.
    deterministic : bool
        Whether to use a deterministic policy.
    """

    def __init__(
        self,
        env: VecEnv,
        model_path: str = "./model_rsl.pt",
        config_path: str = "../rsl_config.yaml",
        device: str = "cuda",
        deterministic: bool = True,
    ):
        self.env = env
        self.deterministic = deterministic
        self.device = device
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_path
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Cannot create agent, policy file '{self.model_path}' not found!"
            )

        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        ) as file:
            config = yaml.safe_load(file)

        train_config = config["train_cfg"]

        runner = OnPolicyRunner(env, train_config, "./", device="cuda")
        runner.load(self.model_path)
        self.policy = runner.get_inference_policy(device=self.device)  # pyright:ignore[reportUnknownMemberType]

    def get_action(self, obs: TensorDict) -> torch.Tensor:
        """
        Get the action from the policy.

        Parameters
        ----------
        obs : TensorDict
            The observation.

        Returns
        -------
        torch.Tensor
            The action.
        """
        with torch.no_grad():
            action = cast(torch.Tensor, self.policy(obs))  # pyright:ignore[reportUnknownMemberType]
        return action
