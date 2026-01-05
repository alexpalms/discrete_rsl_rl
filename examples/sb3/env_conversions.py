"""Environment conversions for Stable Baselines3."""

import logging
import os
import time
from typing import Any, cast

import numpy as np
import torch
from gymnasium import Wrapper
from legged_locomotion_continuous.environment.environment import (
    Environment as EnvironmentContinuous,
)
from maintenance_scheduling_multidiscrete.environment.environment import (
    Environment as EnvironmentMultidiscrete,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs


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
        super().__init__(  # pyright:ignore[reportUnknownMemberType]
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
        return cast(np.ndarray, obs["policy"].cpu().numpy())  # pyright:ignore[reportUnknownMemberType]

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
        new_info = cast(
            list[dict[str, Any]], [info for _ in range(self.vec_env.num_envs)]
        )

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
                    "r": float(round(ep_rew, 6)),
                    "l": float(ep_len),
                    "t": float(round(time.time() - self.t_start, 6)),
                }
                if "episode" not in new_info[idx]:
                    new_info[idx]["episode"] = ep_info
                else:
                    new_info[idx]["episode"].update(ep_info)

                self.rewards[idx] = 0.0
                self.steps[idx] = 0

        self.rewards += rew
        self.steps += np.ones((self.vec_env.num_envs,), dtype=np.int32)

        return cast(np.ndarray, obs["policy"].cpu().numpy()), rew, done, new_info  # pyright:ignore[reportUnknownMemberType]

    def render(self, mode: str | None = None) -> np.ndarray | None:
        """
        Render the environment.

        Parameters
        ----------
        mode : str, optional
            The mode to render the environment.
            The keyword arguments to pass to the environment render method.

        Returns
        -------
        np.ndarray | None
            The rendered environment.
        """
        self.vec_env.render()
        return None

    def close(self) -> None:
        """Close the environment."""
        self.vec_env.close()

    def get_images(self) -> list[Any]:
        """
        Get the images of the environment.

        Returns
        -------
        list[Any]
            The images of the environment.
        """
        return [None for _ in range(self.vec_env.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """
        Get the attribute of the environment.

        Parameters
        ----------
        attr_name : str
            The attribute name.
        indices : VecEnvIndices, optional
            The indices of the environments.

        Returns
        -------
        list[Any]
            The attribute of the environment.
        """
        return [None] * self.vec_env.num_envs

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """
        Set the attribute of the environment.

        Parameters
        ----------
        attr_name : str
            The attribute name.
        value : Any
            The value to set.
        indices : VecEnvIndices, optional
            The indices of the environments.

        Returns
        -------
        None
        """
        pass

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """
        Call the method of the environment.

        Parameters
        ----------
        method_name : str
            The method name.
        *method_args : Any
            The method arguments.
        indices : VecEnvIndices, optional
            The indices of the environments.
        **method_kwargs : Any
            The method keyword arguments.

        Returns
        -------
        list[Any]
            The method of the environment.
        """
        return [None] * self.vec_env.num_envs

    def env_is_wrapped(
        self,
        wrapper_class: type[Wrapper[Any, Any, Any, Any]],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        """
        Check if the environment is wrapped.

        Parameters
        ----------
        wrapper_class : type[Wrapper]
            The wrapper class.
        indices : VecEnvIndices, optional
            The indices of the environments.

        Returns
        -------
        list[bool]
            The wrapped environment.
        """
        return [False] * self.vec_env.num_envs
