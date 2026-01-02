"""Miscellaneous functions for the examples."""

import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sb3.env_conversions import CpuVecEnvToSb3VecEnv  # type: ignore
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from examples.legged_locomotion_continuous.environment.environment import (
    Environment as EnvironmentContinuous,
)
from examples.maintenance_scheduling_multidiscrete.environment.environment import (
    Environment as EnvironmentMultidiscrete,
)


# Make Stable Baselines3 Env function
def make_sb3_env(
    env_class: Callable[..., EnvironmentContinuous | EnvironmentMultidiscrete],
    env_args: dict[str, Any],
    seed: int | None = None,
    monitor_folder: str | None = None,
) -> CpuVecEnvToSb3VecEnv:
    """
    Create a wrapped, monitored VecEnv.

    Parameters
    ----------
    env_class : EnvironmentContinuous | EnvironmentMultidiscrete
        The environment class.
    env_args : dict
        The environment arguments.
    seed : int
        The seed.
    monitor_folder : str, optional
        The monitor folder, by default "/tmp/invai/".

    Returns
    -------
    VecEnv
        The vectorized environment.
    """
    if seed is None:
        seed = int(time.time())

    if monitor_folder is None:
        monitor_folder = tempfile.mkdtemp(prefix="invai_")

    env = env_class(**env_args)
    env = CpuVecEnvToSb3VecEnv(env, monitor_folder=monitor_folder)
    env.reset(seed=seed)
    set_random_seed(seed)

    return env


# Linear scheduler for RL agent parameters
def linear_schedule(
    initial_value: float | str, final_value: float = 0.0
) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Parameters
    ----------
    initial_value : float | str
        The initial value.
    final_value : float, optional
        The final value, by default 0.0.

    Returns
    -------
    Callable[[float], float]
        The linear scheduler.
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        if not initial_value > 0.0:
            raise ValueError(
                "linear_schedule work only with positive decreasing values"
            )

    def func(progress: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        Parameters
        ----------
        progress : float
            The progress.

        Returns
        -------
        float
            The progress.
        """
        return final_value + progress * (initial_value - final_value)

    return func


# AutoSave Callback
class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every `check_freq` steps.

    Parameters
    ----------
    check_freq : int
        The frequency of the saving.
    num_envs : int
        The number of environments.
    save_path : str
        The path to the folder where the model will be saved.
    filename_prefix : str
        The filename prefix.
    starting_steps : int
        The starting steps.
    verbose : int
        The verbosity level.
    """

    def __init__(
        self,
        check_freq: int,
        num_envs: int,
        save_path: str,
        filename_prefix: str = "",
        starting_steps: int = 0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.check_freq = int(check_freq / num_envs)
        self.num_envs = num_envs
        self.save_path_base = Path(save_path)
        self.filename = filename_prefix + "autosave_"
        self.starting_steps = starting_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                self.logger.info(f"Saving latest model to {self.save_path_base}")  # pyright: ignore[reportUnknownMemberType]
            # Save the agent
            self.model.save(
                self.save_path_base
                / (
                    self.filename
                    + str(self.starting_steps + self.n_calls * self.num_envs)
                )
            )

        return True


# Initialize the starting steps number
class StartingSteps(BaseCallback):
    """
    Callback for setting the starting number of steps.

    Parameters
    ----------
    starting_steps : int
        The starting steps.
    """

    def __init__(self, starting_steps: int):
        super().__init__()
        self.starting_steps = starting_steps

    def _init_callback(self) -> None:
        self.model.num_timesteps = self.starting_steps

    def _on_step(self) -> bool:
        return True


# Custom Metrics
class CustomMetrics(BaseCallback):
    """
    Custom callback for logging values from the environment info dictionary.

    Automatically detects and logs all metrics with prefix 'custom_metrics/' in the info dict.

    Parameters
    ----------
    verbose : int
        The verbosity level.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.value_buffer: dict[str, list[Any]] = {}

    def _on_step(self) -> bool:
        # Get info dict from locals
        infos = self.locals["infos"]

        # Collect all metrics from first info dict to initialize buffer if needed
        if len(infos) > 0:
            for key in infos[0].keys():
                if key.startswith("custom_metrics/") and key not in self.value_buffer:
                    self.value_buffer[key] = []

        # Aggregate values across all environments
        for key in self.value_buffer.keys():
            values: list[Any] = []
            for info in infos:
                if key in info:
                    values.append(info[key])

            if values:
                # Calculate mean if we have any values
                mean_value = np.mean(values)
                self.value_buffer[key].append(mean_value)

        # Log values every step
        for key in list(self.value_buffer.keys()):
            if self.value_buffer[key]:
                # Calculate mean over collected values
                mean_value = np.mean(self.value_buffer[key])
                # Log to tensorboard
                self.logger.record(
                    key, mean_value
                )  # key already includes "custom_metrics/" prefix
                # Clear buffer
                self.value_buffer[key] = []

        return True
