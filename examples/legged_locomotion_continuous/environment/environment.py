"""Legged Locomotion Continuous Environment."""

import math
import os
from collections.abc import Callable
from typing import Any, cast

import genesis as gs  # pyright:ignore[reportMissingTypeStubs]
import numpy as np
import torch
import yaml
from genesis.utils.geom import (  # pyright:ignore[reportMissingTypeStubs]
    inv_quat,  # pyright:ignore[reportUnknownVariableType]
    quat_to_xyz,  # pyright:ignore[reportUnknownVariableType]
    transform_by_quat,  # pyright:ignore[reportUnknownVariableType]
    transform_quat_by_quat,  # pyright:ignore[reportUnknownVariableType]
)
from gymnasium import spaces
from tensordict import TensorDict  # pyright:ignore[reportMissingTypeStubs]

from rsl_rl.env.vec_env import VecEnv  # pyright:ignore[reportMissingTypeStubs]


def gs_rand_float(
    lower: float, upper: float, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """
    Generate a random float between lower and upper.

    Parameters
    ----------
    lower : float
        The lower bound.
    upper : float
        The upper bound.
    shape : tuple[int, ...]
        The shape of the tensor.
    device : torch.device
        The device to run the tensor on.

    Returns
    -------
    torch.Tensor
        The random float tensor.
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Environment(VecEnv):
    """
    Legged Locomotion Continuous Environment.

    Parameters
    ----------
    num_envs : int
        The number of environments to run in parallel.
    config_path : str
        The path to the configuration file.
    show_viewer : bool
        Whether to show the viewer.
    """

    def __init__(
        self,
        num_envs: int,
        config_path: str = "./config.yaml",
        show_viewer: bool = False,
    ) -> None:
        local_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(local_path, config_path)) as file:
            config = yaml.safe_load(file)

        env_cfg = config["env_cfg"]
        obs_cfg = config["obs_cfg"]
        reward_cfg = config["reward_cfg"]
        command_cfg = config["command_cfg"]

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = cast(torch.device, gs.device)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_obs,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)  # pyright:ignore[reportUnknownMemberType]

        # names to indices
        self.motors_dof_idx: list[int] = [
            self.robot.get_joint(name).dof_start  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            for name in self.env_cfg["joint_names"]
        ]

        # PD control parameters
        self.robot.set_dofs_kp(  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            [self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx
        )
        self.robot.set_dofs_kv(  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            [self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions: dict[str, Callable[[], torch.Tensor]] = {}
        self.episode_sums: dict[str, torch.Tensor] = {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    def _resample_commands(self, envs_idx: torch.Tensor) -> None:
        low, high = self.command_cfg["lin_vel_x_range"]
        for idx, key in enumerate(["lin_vel_x", "lin_vel_y", "ang_vel"]):
            low, high = self.command_cfg[f"{key}_range"]
            self.commands[envs_idx, idx] = gs_rand_float(
                low, high, (len(envs_idx),), self.device
            )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ]:
        """
        Step the environment.

        Parameters
        ----------
        actions : torch.Tensor
            The actions to step the environment with.

        Returns
        -------
        tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]]]
            The observation, reward, done, and extras.
        """
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = cast(torch.Tensor, self.robot.get_pos())  # pyright:ignore
        self.base_quat[:] = cast(torch.Tensor, self.robot.get_quat())  # pyright:ignore
        self.base_euler = cast(
            torch.Tensor,
            quat_to_xyz(
                transform_quat_by_quat(
                    torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                    self.base_quat,
                ),
                rpy=True,
                degrees=True,
            ),
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = cast(
            torch.Tensor,
            transform_by_quat(self.robot.get_vel(), inv_base_quat),  # pyright:ignore
        )
        self.base_ang_vel[:] = cast(
            torch.Tensor,
            transform_by_quat(self.robot.get_ang(), inv_base_quat),  # pyright:ignore
        )
        self.projected_gravity = cast(
            torch.Tensor, transform_by_quat(self.global_gravity, inv_base_quat)
        )
        self.dof_pos[:] = cast(
            torch.Tensor,
            self.robot.get_dofs_position(self.motors_dof_idx),  # pyright:ignore
        )
        self.dof_vel[:] = cast(
            torch.Tensor,
            self.robot.get_dofs_velocity(self.motors_dof_idx),  # pyright:ignore
        )

        # resample commands
        envs_idx = (
            (
                self.episode_length_buf
                % int(self.env_cfg["resampling_time_s"] / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = torch.Tensor([1.0])

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = torch.tensor([0.0])
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"] = {"critic": self.obs_buf}

        return (
            TensorDict({"policy": self.obs_buf}),
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def render(self) -> None:
        """Render the environment."""
        return None

    def get_observations(self) -> TensorDict:
        """
        Get the observations.

        Returns
        -------
        TensorDict
            The observations.
        """
        return TensorDict({"policy": self.obs_buf})

    def get_privileged_observations(self) -> None:
        """
        Get the privileged observations.

        Returns
        -------
        None
        """
        return None

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """
        Reset the environment.

        Parameters
        ----------
        envs_idx : torch.Tensor
            The indices of the environments to reset.
        """
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(  # pyright:ignore
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(  # pyright:ignore
            self.base_pos[envs_idx],
            zero_velocity=False,  # pyright:ignore
            envs_idx=envs_idx,  # pyright:ignore
        )
        self.robot.set_quat(  # pyright:ignore
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType]

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        episode_dict: dict[str, torch.Tensor] = {}
        for key in self.episode_sums.keys():
            episode_dict["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.extras["episode"] = episode_dict
        self._resample_commands(envs_idx)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[TensorDict, None]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int | None
            The seed to reset the environment with.
        options : dict[str, Any] | None
            The options to reset the environment with.

        Returns
        -------
        tuple[TensorDict, None]
            The observation and the extras.
        """
        self.scene.reset()  # pyright:ignore[reportUnknownMemberType]
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return TensorDict({"policy": self.obs_buf}), None

    def close(self) -> None:
        """Close the environment."""
        pass

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self) -> torch.Tensor:
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self) -> torch.Tensor:
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self) -> torch.Tensor:
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self) -> torch.Tensor:
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
