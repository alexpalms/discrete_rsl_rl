# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Wandb is required to log to Weights and Biases."
    ) from ModuleNotFoundError


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict[str, Any]):
        super().__init__(log_dir, flush_secs=flush_secs)  # pyright:ignore[reportUnknownMemberType]

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError(
                "Please specify wandb_project in the runner config, e.g. legged_gym."
            ) from KeyError

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})  # pyright:ignore[reportUnkownMemberType]

        self.name_map: dict[str, str] = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(
        self,
        env_cfg: dict[str, Any],
        runner_cfg: dict[str, Any],
        alg_cfg: dict[str, Any],
        policy_cfg: dict[str, Any],
    ):
        wandb.config.update({"runner_cfg": runner_cfg})  # pyright:ignore[reportUnkownMemberType]
        wandb.config.update({"policy_cfg": policy_cfg})  # pyright:ignore[reportUnkownMemberType]
        wandb.config.update({"alg_cfg": alg_cfg})  # pyright:ignore[reportUnkownMemberType]
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})  # pyright:ignore[reportUnkownMemberType]
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})  # pyright:ignore[reportUnkownMemberType]

    def add_scalar(
        self,
        tag: Any,
        scalar_value: Any,
        global_step: int | None = None,
        walltime: Any | None = None,
        new_style: bool = False,
        double_precision: bool = False,
    ) -> None:
        super().add_scalar(  # pyright:ignore[reportUnkownMemberType]
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
            double_precision=double_precision,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(
        self,
        env_cfg: dict[str, Any],
        runner_cfg: dict[str, Any],
        alg_cfg: dict[str, Any],
        policy_cfg: dict[str, Any],
    ):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, _iter: int | None):
        wandb.save(model_path, base_path=os.path.dirname(model_path))  # pyright:ignore[reportUnknownMemberType]

    def save_file(self, path: str, _iter: int | None = None):
        wandb.save(path, base_path=os.path.dirname(path))  # pyright:ignore[reportUnknownMemberType]

    """
    Private methods.
    """

    def _map_path(self, path: str):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
