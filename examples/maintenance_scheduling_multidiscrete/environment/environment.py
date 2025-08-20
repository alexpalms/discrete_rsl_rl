import numpy as np
from gymnasium import spaces
import gymnasium as gym
import torch
from tensordict import TensorDict
import json
import os

class Environment(gym.Env):
    def __init__(self, num_envs=2048, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.simulation_duration = 365 # Days

        local_path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(local_path, "component_structure.json"), "r") as f:
            self.network = json.load(f)

        self.n = len(self.network["components_points"])
        self.connectivity_map = torch.tensor(np.array(self.network["connectivity"]), dtype=torch.float32, device=self.device)
        self.components_points_values = np.sqrt(np.sum(np.array(self.network["connectivity"]), axis=1))

        # Intrinsic component failure probability (daily)
        self.mtbf = 150.0
        self.k = 1.2

        # Probability of failure propagating between connected components
        connected_component_failure_propagation_probability = 0.15

        self.render_mode = None

        self.components_points = torch.tensor(self.components_points_values, dtype=torch.float32, device=self.device)
        self.components_points = self.components_points.repeat(num_envs, 1)
        self.components_failure_probability = torch.zeros((num_envs, self.n), dtype=torch.float32, device=self.device)
        self.delta_probability_prev = torch.zeros((num_envs, self.n), dtype=torch.float32, device=self.device)
        self.component_age = torch.ones((num_envs, self.n), dtype=torch.float32, device=self.device)

        self.components_intrinsic_failure_probability = torch.zeros((num_envs, self.n), dtype=torch.float32, device=self.device)
        self.components_points /= self.components_points.sum(dim=1, keepdim=True)

        self.c2c_failure_propagation_probability = self.connectivity_map * connected_component_failure_propagation_probability
        self.no_cleaning_prob = torch.empty_like(self.components_failure_probability)
        self.mask = torch.zeros((num_envs, self.n), dtype=torch.bool, device=self.device)
        self.one = torch.tensor(1.0, device=self.device)

        self.actions = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        self.obs_buf = torch.zeros((self.num_envs, 2 * self.n), device=self.device, dtype=torch.float32)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.cumulative_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.done_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self.extras = dict()

        # Observations are dictionaries
        # Concatenate the points (The worth of each component) and the failure probability
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2*self.n, ), dtype=np.float32)

        self.num_actions = [self.n + 1]
        self.action_space = spaces.MultiDiscrete(self.num_actions)
        self.clock_step = 0

    def update_probabilities_to_next_day(self):
        # Update failure probabilities
        no_failures = self.one - self.components_failure_probability

        # Intrinsic failure probability delta
        h = (self.k / self.mtbf) * (self.component_age / self.mtbf) ** (self.k - 1)
        self.components_intrinsic_failure_probability = self.one - torch.exp(-h)
        intrinsic_delta = no_failures * self.components_intrinsic_failure_probability

        # Probability of NOT failing because of a neighbor failure
        # (1 - i_t[a] * T[a,b]) for each a,b
        no_failure_from_neighbor = self.one - (self.delta_probability_prev.unsqueeze(1) * self.c2c_failure_propagation_probability[None, :, :])

        neighbor_delta = no_failures * (self.one - torch.exp(torch.sum(torch.log(no_failure_from_neighbor), dim=1)))

        # Product over all sources for each target
        self.delta_probability_prev = -self.components_failure_probability
        self.components_failure_probability = self.one - no_failures * (self.one - intrinsic_delta) * (self.one - neighbor_delta)
        self.delta_probability_prev += self.components_failure_probability

        self.component_age += self.one

    def maintenance(self, actions):
        actions_shifted = actions - self.one
        valid_mask = actions_shifted >= 0

        # Make invalid actions a safe index
        safe_actions = torch.where(valid_mask, actions_shifted, torch.zeros_like(actions_shifted)).to(torch.int)

        # Create index grid for batch dimension
        env_ids = torch.arange(self.num_envs, device=self.device)[:, None].expand_as(safe_actions)

        # Scatter all at once
        self.mask.fill_(False)
        self.mask[env_ids, safe_actions] = valid_mask

        # Cleaning probability adjustment
        self.components_failure_probability[self.mask] *= 0.0
        self.component_age[self.mask] *= 0.0
        self.delta_probability_prev[self.mask] = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super(type(self), self).reset(seed=seed)
        self.clock_step = 0
        self.components_failure_probability.zero_()
        self.delta_probability_prev.zero_()
        self.component_age.fill_(1.0)
        self.update_probabilities_to_next_day()
        self._compute_observation()
        self.done_buf.fill_(False)
        self.extras["log"] = {"average_cumulative_reward": -torch.mean(self.cumulative_rew_buf)}
        self.cumulative_rew_buf.zero_()

        return TensorDict({"policy": self.obs_buf}), None

    def step(self, actions):
        self.clock_step += 1

        self.maintenance(actions)

        self.update_probabilities_to_next_day()

        # Get the next observation, reward and
        if self.clock_step >= self.simulation_duration:
            self.reset()
            self.done_buf.fill_(True)
        else:
            self.done_buf.fill_(False)

        # Reward
        self.rew_buf = self.cumulative_rew_buf
        self.cumulative_rew_buf = torch.matmul(self.components_failure_probability, self.components_points[0, :])
        self.rew_buf -= self.cumulative_rew_buf

        # Observations
        self._compute_observation()

        return TensorDict({"policy": self.obs_buf}), self.rew_buf, self.done_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf})

    def _compute_observation(self):
        self.obs_buf = torch.cat(
            [
                self.components_points,
                self.components_failure_probability,
            ],
            dim=-1,
        )
        return self.obs_buf

    def render(self):
        if self.render_mode == "rgb_array":
            raise NotImplementedError()
        return None

    def close(self):
        pass