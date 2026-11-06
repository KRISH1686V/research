"""HER replay as preallocated GPU tensors. Relabelling and reward recomputation are one batched call.

Follows SB3's HerReplayBuffer: per-env ring of capacity // n_envs rows, only completed episodes are sampled, `future`
goal selection, her_ratio = 1 - 1 / (k + 1), virtual transitions first. No infos are stored; the five scalars the reward
needs are kept instead. Rewards match compute_reward_batch for reward_mode "track" without budgets (tests/test_fast_learner.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RewardConfig:
    sigma_v: float
    sigma_w: float
    fall_penalty: float
    progress_weight: float
    gait_shaping: bool
    smooth_penalties: bool

    @classmethod
    def from_env(cls, env) -> "RewardConfig":
        if env.reward_mode != "track" or env.budget_goal_enabled:
            raise ValueError("fast replay supports reward_mode 'track' without budgets")
        return cls(float(env.sigma_v), float(env.sigma_w), float(env.fall_penalty), float(env.progress_weight),
                   bool(env.gait_shaping), bool(env.smooth_penalties))


def relabel_reward(cfg: RewardConfig, achieved: torch.Tensor, goal: torch.Tensor, r_smooth: torch.Tensor,
                   r_gait_move: torch.Tensor, r_gait_still: torch.Tensor, r_flight: torch.Tensor,
                   fell: torch.Tensor) -> torch.Tensor:
    """compute_reward_batch on tensors. `achieved` and `goal` are [vx, vy, wz]."""
    v = achieved[:, :2] - goal[:, :2]
    w = achieved[:, 2] - goal[:, 2]
    r = (-(v * v).sum(-1) / (2.0 * cfg.sigma_v ** 2)).exp() * (-(w * w) / (2.0 * cfg.sigma_w ** 2)).exp()
    if cfg.progress_weight:
        progress = (achieved[:, 0] / goal[:, 0]).clamp(0.0, 1.0)
        r = r + cfg.progress_weight * torch.where(goal[:, 0] > 0.1, progress, torch.zeros_like(r))
    if cfg.smooth_penalties:
        r = r + r_smooth
    if cfg.gait_shaping:
        moving = goal[:, :2].norm(dim=-1) > 0.1
        r = r + r_flight + torch.where(moving, r_gait_move, r_gait_still)
    return torch.where(fell > 0.5, torch.full_like(r, -cfg.fall_penalty), r)


class GpuHerReplay:
    """Row layout: obs, next_obs, action, reward, done, r_smooth, r_gait_move, r_gait_still, r_flight, fell."""

    N_SCALARS = 7

    def __init__(self, capacity: int, n_envs: int, batch_size: int, n_sampled_goal: int, obs_dim: int, act_dim: int,
                 reward_cfg: RewardConfig, device: torch.device, achieved_slice=(0, 3), desired_slice=(3, 6),
                 n_step: int = 1, gamma: float = 0.99, priv_dim: int = 0, mirror: dict | None = None):
        self.rows, self.n_envs, self.batch_size = max(capacity // n_envs, 1), n_envs, batch_size
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.gamma_pow = torch.tensor([gamma ** k for k in range(self.n_step + 1)], dtype=torch.float32, device=device)
        self.steps = torch.arange(self.n_step, device=device)
        self.batch_index = torch.arange(batch_size, device=device)
        self.obs_dim, self.act_dim, self.cfg, self.device = obs_dim, act_dim, reward_cfg, device
        self.nb_virtual = int((1.0 - 1.0 / (n_sampled_goal + 1)) * batch_size)
        size = self.rows * n_envs
        self.i_next = obs_dim
        self.i_act = 2 * obs_dim
        self.i_rew = self.i_act + act_dim
        self.priv_dim = int(priv_dim)
        self.i_priv = self.i_rew + self.N_SCALARS
        self.i_next_priv = self.i_priv + self.priv_dim
        self.width = self.i_next_priv + self.priv_dim
        self.mirror = None
        if mirror:
            as_t = lambda x, kind: torch.as_tensor(x, device=device, dtype=kind)
            self.mirror = {k: as_t(v, torch.int64 if k.endswith("perm") else torch.float32) for k, v in mirror.items()}
        self.a0, self.a1 = achieved_slice
        self.d0, self.d1 = desired_slice
        self.store = torch.zeros((size, self.width), dtype=torch.float32, device=device)
        self.ep_start = torch.zeros(size, dtype=torch.int64, device=device)
        self.ep_len = torch.zeros(size, dtype=torch.int64, device=device)
        self.valid = torch.zeros(size, dtype=torch.int64, device=device)
        self.n_valid = torch.zeros((), dtype=torch.float32, device=device)
        self._n_valid = 0
        self.pos = 0
        self.env_index = torch.arange(n_envs, device=device)
        self._ep_len_np = np.zeros((self.rows, n_envs), dtype=np.int64)
        self._ep_start_np = np.zeros((self.rows, n_envs), dtype=np.int64)
        self._current_start = np.zeros(n_envs, dtype=np.int64)

    def add(self, rows: torch.Tensor, ended: np.ndarray) -> None:
        """Store one vector step. `rows` is (n_envs, width); `ended` marks terminated or truncated envs."""
        invalidated = self._forget_overwritten()
        self.store[self.pos * self.n_envs + self.env_index] = rows
        self._ep_start_np[self.pos] = self._current_start
        pos = self.pos
        self.pos = (self.pos + 1) % self.rows
        if invalidated:
            self._rebuild_valid()
        for env in np.flatnonzero(ended):
            self._finish_episode(int(env), pos)
            self._current_start[env] = self.pos

    def _forget_overwritten(self) -> bool:
        lengths = self._ep_len_np[self.pos]
        if not lengths.any():
            return False
        for env in np.flatnonzero(lengths):
            start, length = self._ep_start_np[self.pos, env], lengths[env]
            rows = np.arange(self.pos, start + length) % self.rows
            self._ep_len_np[rows, env] = 0
        return True

    def _finish_episode(self, env: int, last_row: int) -> None:
        start = int(self._current_start[env])
        end = last_row + 1 if last_row >= start else last_row + 1 + self.rows
        rows = np.arange(start, end) % self.rows
        self._ep_len_np[rows, env] = end - start
        flat = torch.as_tensor(rows * self.n_envs + env, device=self.device)
        self.ep_start[flat] = start
        self.ep_len[flat] = end - start
        self.valid[self._n_valid:self._n_valid + len(rows)] = flat
        self._n_valid += len(rows)
        self.n_valid.fill_(float(self._n_valid))

    def _rebuild_valid(self) -> None:
        valid = np.flatnonzero(self._ep_len_np.ravel())
        self._n_valid = len(valid)
        self.valid[:self._n_valid] = torch.as_tensor(valid, device=self.device)
        self.n_valid.fill_(float(self._n_valid))

    @property
    def ready(self) -> bool:
        return self._n_valid > 0

    def sample(self):
        """Returns obs, action, reward, next_obs, done, discount. Static shapes and no data-dependent Python, so it can be graph-captured."""
        if self.n_step > 1:
            return self._sample_n_step()
        d = self.device
        pick = self.valid[(torch.rand(self.batch_size, device=d) * self.n_valid).long()]
        b = self.store[pick]
        V = self.nb_virtual
        pv = pick[:V]
        env = pv % self.n_envs
        row = torch.div(pv, self.n_envs, rounding_mode="floor")
        start, length = self.ep_start[pv], self.ep_len[pv]
        cur = (row - start) % self.rows
        fut = cur + (torch.rand(V, device=d) * (length - cur).float()).long()
        goal_rows = (fut + start) % self.rows
        goal = self.store[goal_rows * self.n_envs + env][:, self.i_next + self.a0:self.i_next + self.a1]
        b[:V, self.d0:self.d1] = goal
        b[:V, self.i_next + self.d0:self.i_next + self.d1] = goal
        i = self.i_rew
        b[:V, i] = relabel_reward(self.cfg, b[:V, self.i_next + self.a0:self.i_next + self.a1], goal,
                                  b[:V, i + 2], b[:V, i + 3], b[:V, i + 4], b[:V, i + 5], b[:V, i + 6])
        return self._finish(b[:, :self.obs_dim], b[:, self.i_act:self.i_rew], b[:, i], b[:, self.i_next:self.i_act], b[:, i + 1], self.gamma_pow[1], b)

    def _finish(self, obs, act, rew, nxt, done, disc, b, final=None):
        """Attach critic-only state and apply mirroring. Stays a 6-tuple when neither is in use."""
        priv = b[:, self.i_priv:self.i_next_priv] if self.priv_dim else None
        next_priv = (final if final is not None else b)[:, self.i_next_priv:self.i_next_priv + self.priv_dim] if self.priv_dim else None
        if self.mirror is not None:
            m = self.mirror
            flip = (torch.rand(self.batch_size, device=self.device) < 0.5)[:, None]
            obs = torch.where(flip, obs[:, m["obs_perm"]] * m["obs_sign"], obs)
            nxt = torch.where(flip, nxt[:, m["obs_perm"]] * m["obs_sign"], nxt)
            act = torch.where(flip, act[:, m["act_perm"]], act)
            if self.priv_dim:
                priv = torch.where(flip, priv[:, m["priv_perm"]] * m["priv_sign"], priv)
                next_priv = torch.where(flip, next_priv[:, m["priv_perm"]] * m["priv_sign"], next_priv)
        if self.priv_dim:
            return obs, act, rew, nxt, done, disc, priv, next_priv
        return obs, act, rew, nxt, done, disc

    def _sample_n_step(self):
        """Up to n_step rewards from the sampled transition onward, stopping at the episode end.

        Virtual transitions recompute every reward in the window against the relabelled goal. No off-policy correction.
        """
        d, B, V, E, i = self.device, self.batch_size, self.nb_virtual, self.n_envs, self.i_rew
        pick = self.valid[(torch.rand(B, device=d) * self.n_valid).long()]
        b = self.store[pick]
        env = pick % E
        row = torch.div(pick, E, rounding_mode="floor")
        start, length = self.ep_start[pick], self.ep_len[pick]
        cur = (row - start) % self.rows
        fut = cur[:V] + (torch.rand(V, device=d) * (length[:V] - cur[:V]).float()).long()
        goal_rows = (fut + start[:V]) % self.rows
        goal = self.store[goal_rows * E + env[:V]][:, self.i_next + self.a0:self.i_next + self.a1]

        k = cur[:, None] + self.steps
        inside = k < length[:, None]
        seq_rows = (start[:, None] + torch.minimum(k, length[:, None] - 1)) % self.rows
        seq = self.store[seq_rows * E + env[:, None]]
        rewards = seq[:, :, i].clone()
        n = self.n_step
        flat = seq[:V].reshape(V * n, self.width)
        rewards[:V] = relabel_reward(self.cfg, flat[:, self.i_next + self.a0:self.i_next + self.a1], goal.repeat_interleave(n, dim=0),
                                     flat[:, i + 2], flat[:, i + 3], flat[:, i + 4], flat[:, i + 5], flat[:, i + 6]).reshape(V, n)
        reward = (rewards * inside.float() * self.gamma_pow[:n]).sum(1)
        last = inside.sum(1) - 1
        final = seq[self.batch_index, last]
        next_obs = final[:, self.i_next:self.i_act].clone()
        b[:V, self.d0:self.d1] = goal
        next_obs[:V, self.d0:self.d1] = goal
        return self._finish(b[:, :self.obs_dim], b[:, self.i_act:self.i_rew], reward, next_obs, final[:, i + 1], self.gamma_pow[last + 1], b, final)
