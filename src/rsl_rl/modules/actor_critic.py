# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        action_type: str = "continuous",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # store action type
        assert action_type in ("continuous", "multi_discrete"), "action_type must be 'continuous' or 'multi_discrete'"
        self.action_type = action_type

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # determine action paramization
        # continuous: num_actions is int (action dim)
        # multi_discrete: num_actions should be list/tuple of ints (num categories per branch)
        if self.action_type == "continuous":
            assert isinstance(num_actions, int), "For continuous action_type, num_actions must be int"
            self.num_actions = num_actions
            actor_output_dim = num_actions
            # Action noise parameters
            self.noise_std_type = noise_std_type
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:  # multi_discrete
            if isinstance(num_actions, int):
                self.num_actions_list = list([num_actions])
            elif isinstance(num_actions, tuple):
                self.num_actions_list = list(num_actions)
            else:
                assert isinstance(num_actions, list), "For multi_discrete action_type, num_actions must be int/tuple (automatically converted here in list) or list"
                self.num_actions_list = num_actions
            self.num_branches = len(self.num_actions_list)
            actor_output_dim = int(sum(self.num_actions_list))  # we will output logits concatenated
            # std/log_std are not used for discrete
            self.noise_std_type = None
            self.std = None
            self.log_std = None

        # actor
        self.actor = MLP(num_actor_obs, actor_output_dim, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # For discrete, store latest logits split per branch
        self._last_logits = None

        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        # for continuous: mean; for discrete: return logits (convention)
        if self.action_type == "continuous":
            return self.distribution.mean
        else:
            # return concatenated logits (raw), this mirrors storage usage older code expects a tensor
            return torch.cat(self._last_logits, dim=-1)

    @property
    def action_std(self):
        if self.action_type == "continuous":
            return self.distribution.stddev
        else:
            # return zeros shaped like concatenated logits
            return torch.zeros_like(self.action_mean)

    @property
    def entropy(self):
        if self.action_type == "continuous":
            # gaussian: sum over action dims -> shape [B]
            return self.distribution.entropy().sum(dim=-1)
        else:
            # categorical list: each cat.entropy() -> [B]
            # stack -> [B, num_branches], sum over branch dim -> [B]
            entropies = torch.stack([cat.entropy() for cat in self.distribution], dim=1)
            return entropies.sum(dim=1)

    def update_distribution(self, obs):
        if self.action_type == "continuous":
            # compute mean
            mean = self.actor(obs)
            # compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            # create distribution
            self.distribution = Normal(mean, std)
            self._last_logits = None
        else:
            # discrete: produce logits and split per branch
            logits = self.actor(obs)  # shape: [B, sum(num_actions_list)]
            # split logits per branch
            splits = torch.split(logits, self.num_actions_list, dim=-1)
            cats = []
            for l in splits:
                cats.append(Categorical(logits=l))
            self.distribution = cats  # list of Categorical objects
            self._last_logits = splits

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        if self.action_type == "continuous":
            return self.distribution.sample()
        else:
            # sample each categorical -> produce int per branch
            samples = [cat.sample().unsqueeze(-1) for cat in self.distribution]
            return torch.cat(samples, dim=-1).to(dtype=torch.long)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        out = self.actor(obs)
        if self.action_type == "continuous":
            return out  # mean
        else:
            # return argmax per branch as deterministic action
            splits = torch.split(out, self.num_actions_list, dim=-1)
            # Sample from categorical distributions instead of argmax
            actions = []
            for s in splits:
                # Convert logits to probabilities and sample
                probs = torch.softmax(s, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                actions.append(action.unsqueeze(-1))
            return torch.cat(actions, dim=-1).to(dtype=torch.long)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        """
        actions:
          - continuous -> tensor float [B, action_dim]
          - multi_discrete -> tensor long  [B, num_branches]
        Returns:
          - log prob per sample: shape [B]
        """
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        else:
            # compute per-branch log-probs -> [B, num_branches], sum over branches -> [B]
            logps = torch.stack(
                [cat.log_prob(actions[..., i]) for i, cat in enumerate(self.distribution)], dim=1
            )
            return logps.sum(dim=1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
