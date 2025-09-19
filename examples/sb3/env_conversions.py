"""Environment conversions for Stable Baselines3."""

import logging
import os
import time
from typing import Any

import numpy as np
import torch
from legged_locomotion_continuous.environment.environment import (
    Environment as EnvironmentContinuous,
)
from maintenance_scheduling_multidiscrete.environment.environment import (
    Environment as EnvironmentMultidiscrete,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class CpuVecEnvToSb3VecEnv(VecEnv):
    """
    Convert a CPU-based VecEnv to a Stable Baselines3 VecEnv.

    Parameters
    ----------
    vec_env : VecEnv
        The CPU-based VecEnv to convert.
    monitor_folder : str
        The folder to save the monitor files.
    """

    def __init__(
        self,
        vec_env: EnvironmentMultidiscrete | EnvironmentContinuous,
        monitor_folder: str,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.vec_env = vec_env
        super().__init__(
            num_envs=self.vec_env.num_envs,
            observation_space=self.vec_env.observation_space,
            action_space=self.vec_env.action_space,
        )
        self.t_start = time.time()
        self.monitor_folder = monitor_folder

        # Create subfolders for each environment
        if monitor_folder:
            for i in range(self.vec_env.num_envs):
                env_folder = os.path.join(monitor_folder, str(i))
                os.makedirs(env_folder, exist_ok=True)
                # Monitor
                monitor_file = os.path.join(env_folder, "monitor.csv")
                with open(monitor_file, "w") as f:
                    f.write(f'#{{"t_start": {time.time()}, "env_id": "None"}}\n')
                    f.write("r,l,t\n")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> VecEnvObs:
        """
        Reset the environment.

        Parameters
        ----------
        **kwargs : dict
            The keyword arguments to pass to the environment reset method.

        Returns
        -------
        np.ndarray
            The observation of the environment.
        """
        self.rewards = np.zeros((self.vec_env.num_envs,), dtype=np.float32)
        self.steps = np.zeros((self.vec_env.num_envs,), dtype=np.int32)
        obs, _ = self.vec_env.reset(seed=seed, options=options)
        return obs["policy"].cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        """
        Set the actions for the environment.

        Parameters
        ----------
        actions : np.ndarray
            The actions to set for the environment.
        """
        self.actions = torch.as_tensor(actions, device=self.vec_env.device)

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Step the environment.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]
            The observation, reward, done, and info of the environment.
        """
        obs, rew, done, info = self.vec_env.step(self.actions)
        rew = rew.cpu().numpy()
        done = done.cpu().numpy()
        info = [info for _ in range(self.vec_env.num_envs)]

        # Info & Monitor
        for idx in range(self.vec_env.num_envs):
            if done[idx]:
                # Monitor
                monitor_file = os.path.join(
                    self.monitor_folder, str(idx), "monitor.csv"
                )
                with open(monitor_file, "a") as f:
                    ep_rew = self.rewards[idx]
                    ep_len = self.steps[idx]
                    f.write(
                        f"{round(ep_rew, 6)},{ep_len},{round(time.time() - self.t_start, 6)}\n"
                    )

                # Info
                ep_rew = self.rewards[idx]
                ep_len = self.steps[idx]
                ep_info = {
                    "r": round(ep_rew, 6),
                    "l": ep_len,
                    "t": round(time.time() - self.t_start, 6),
                }
                if "episode" not in info[idx]:
                    info[idx]["episode"] = ep_info
                else:
                    info[idx]["episode"].update(ep_info)

                self.rewards[idx] = 0.0
                self.steps[idx] = 0

        self.rewards += rew
        self.steps += np.ones((self.vec_env.num_envs,), dtype=np.int32)

        return obs["policy"].cpu().numpy(), rew, done, info

    def render(self, **kwargs: dict[str, Any]) -> Any:
        """
        Render the environment.

        Parameters
        ----------
        **kwargs : dict
            The keyword arguments to pass to the environment render method.

        Returns
        -------
        Any
            The rendered environment.
        """
        return self.vec_env.render(**kwargs)

    def close(self) -> None:
        """
        Close the environment.
        """
        self.vec_env.close()

    def get_images(self):
        return [None for _ in range(self.vec_env.num_envs)]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [None] * self.vec_env.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> list[Any]:
        return [None] * self.vec_env.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.vec_env.num_envs
