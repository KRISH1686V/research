"""Opt-in budget-aware adaptation of SB3's installed HER replay buffer."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from BipedRobot.envs.rewards.shared_reward import compute_reward_batch


class BudgetHerReplayBuffer(HerReplayBuffer):
    """Preserve velocity HER and optionally relabel future budget components."""

    def __init__(
        self,
        *args: Any,
        p_budget_relabel: float = 0.6,
        min_budget: np.ndarray | list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.p_budget_relabel = float(p_budget_relabel)
        self.min_budget = (
            np.asarray(min_budget, dtype=np.float32) if min_budget is not None else None
        )

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        if self.p_budget_relabel <= 0.0:
            return super()._get_virtual_samples(batch_indices, env_indices, env)

        obs = {
            key: value[batch_indices, env_indices, :].copy()
            for key, value in self.observations.items()
        }
        next_obs = {
            key: value[batch_indices, env_indices, :].copy()
            for key, value in self.next_observations.items()
        }
        infos = (
            copy.deepcopy(self.infos[batch_indices, env_indices])
            if self.copy_info_dict
            else [{} for _ in range(len(batch_indices))]
        )
        new_goals = self._sample_goals(batch_indices, env_indices)
        budget_mask = (
            (np.random.random(len(batch_indices)) < self.p_budget_relabel)
            & (new_goals.shape[1] > 3)
        )
        if np.any(budget_mask):
            budget_values = np.maximum(new_goals[budget_mask, 3:], 0.0)
            if self.min_budget is not None:
                budget_values = np.maximum(budget_values, self.min_budget)
            new_goals[budget_mask, 3:] = np.clip(budget_values, 0.0, 1.0)
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals

        assert self.env is not None
        env_obj = getattr(self.env, "envs", [self.env])[0]
        reward_mode = getattr(env_obj, "reward_mode", "track")
        budget_goal_enabled = bool(getattr(env_obj, "budget_goal_enabled", False))
        fall_penalty = float(getattr(env_obj, "fall_penalty", 5.0))
        sigma_v = float(getattr(env_obj, "sigma_v", 0.35))
        sigma_w = float(getattr(env_obj, "sigma_w", 0.35))
        rewards = compute_reward_batch(
            achieved_goal=next_obs["achieved_goal"],
            desired_goal=new_goals,
            info=infos,
            reward_mode=reward_mode,
            budget_goal_enabled=budget_goal_enabled,
            fall_penalty=fall_penalty,
            sigma_v=sigma_v,
            sigma_w=sigma_w,
            stand_safe_weights=getattr(env_obj, "stand_safe_weights", None),
        ).astype(np.float32)
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        return DictReplayBufferSamples(
            observations={key: self.to_torch(value) for key, value in obs.items()},
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations={
                key: self.to_torch(value) for key, value in next_obs.items()
            },
            dones=self.to_torch(
                self.dones[batch_indices, env_indices]
                * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self._normalize_reward(rewards.reshape(-1, 1), env)
            ),
        )
