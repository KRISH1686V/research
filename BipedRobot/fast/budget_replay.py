"""GPU replay for token-bucket budgets. Relabels the command and the budget together and recomputes bucket levels exactly.

PROVISIONAL: see budget_env.py. Relabeled budgets per cost: keep the episode's budget (p_keep), the mean cost actually spent up to the
future step (p_achieved), or a fresh draw from the prior. Levels use the closed form of the bucket:
    F(k) = S(k) - r * k,   d(t) = F(t) - min[k <= t] F(k),   b(t) = B - d(t)
where S(k) is the episode's cumulative cost before step k. Static shapes and no data-dependent Python, so it can be graph-captured.
With budget["window_steps"] = W the minimum looks back at most W steps, as in budget_env.py. With budget["mode"] = "episode" the deficit is S(t) and the capacity r * L.

budget["lagrange"] is the dual-ascent comparison: fixed cost limits instead of budgets as goals. Every sampled reward pays lambda * cost, and dual_step raises or lowers lambda
from the costs the policy is spending. Use it with budgets switched off (rates above any possible cost), so only the command is relabeled.
"""
from __future__ import annotations

import numpy as np
import torch

from BipedRobot.fast.replay import GpuHerReplay, RewardConfig, relabel_reward


class BudgetHerReplay(GpuHerReplay):
    N_SCALARS = 10  # reward, done, r_smooth, r_gait_move, r_gait_still, r_flight, fell, cost x 3

    def __init__(self, capacity: int, n_envs: int, batch_size: int, n_sampled_goal: int, obs_dim: int, act_dim: int, reward_cfg: RewardConfig,
                 device: torch.device, budget: dict, max_episode_steps: int = 500, priv_dim: int = 0, mirror: dict | None = None, relabel_keep: tuple = ()):
        # relabel_keep (her.relabel_dims, 6 Nov 2026): velocity-goal dims [vx, vy, wz] that keep the stored command instead of the future achieved one
        self.keep_cmd = torch.tensor([i in relabel_keep for i in range(3)], device=device) if relabel_keep else None
        self.n_cost = len(budget.get("costs", ("flight", "slip", "torque")))
        self.N_SCALARS = 7 + self.n_cost  # before the base class sizes rows from it
        super().__init__(capacity, n_envs, batch_size, n_sampled_goal, obs_dim, act_dim, reward_cfg, device, achieved_slice=(0, 3), desired_slice=(3, 6 + self.n_cost), priv_dim=priv_dim, mirror=mirror)
        self.rate_min = torch.tensor(budget["rate_min"], dtype=torch.float64, device=device)
        self.rate_max = torch.tensor(budget["rate_max"], dtype=torch.float64, device=device)
        self.burst = float(budget["burst_steps"])
        self.p_keep = float(budget["p_keep"])
        self.p_achieved = float(budget["p_achieved"])
        self.alive_bonus = float(budget.get("alive_bonus", 0.0))
        self.p_corner = float(budget.get("p_corner", 0.0))  # as budget_env.py: prior draws that tighten one cost only
        self.cap_floor = torch.as_tensor(np.asarray(budget.get("capacity_floor", [0.0] * self.n_cost), dtype=np.float64), device=device)  # as budget_env.py
        self.soft_overdraft = budget.get("overdraft", "hard") == "soft"  # as budget_env.py: scale, do not zero, the task reward while overdrawn
        self.u_floor = torch.full((), float(budget.get("u_low", 0.0)), dtype=torch.float64, device=device)  # prior draws from [u_floor, 1]; u_low as budget_env.py
        self.L = int(max_episode_steps)
        self.i_cost = self.i_rew + 7
        self.i_level = obs_dim - self.n_cost
        self.cum = torch.zeros((self.rows * n_envs, self.n_cost), dtype=torch.float64, device=device)
        self._cum_np = np.zeros((n_envs, self.n_cost), dtype=np.float64)
        self._cum_row = torch.zeros((n_envs, self.n_cost), dtype=torch.float64, device=device)
        self.steps_L = torch.arange(self.L, device=device)
        self.steps_f = self.steps_L.to(torch.float64)
        self.window = budget.get("window_steps")
        self.steps_back = torch.arange(int(self.window) + 1, device=device) if self.window else None
        self.episode_budget = budget.get("mode", "bucket") == "episode"
        self.span = float(self.L) if self.episode_budget else self.burst
        self.lagrange = budget.get("lagrange")
        if self.lagrange:
            self.lam = torch.full((self.n_cost,), float(self.lagrange.get("init", 0.0)), dtype=torch.float32, device=device)
            self._lam_np = np.full(self.n_cost, float(self.lagrange.get("init", 0.0)))
            self._limit = np.asarray(self.lagrange["limit"], dtype=np.float64)
            # lagrange.normalize (11 Nov 2026, default off): each cost is paid per unit of its own limit, lambda * cost / limit, so costs of very different
            # sizes (flight about 1e-3 a step, joint shake about 3e-5) get multipliers of the same order. With lr 0 it is a fixed penalty of init per limit.
            self.lag_scale = torch.as_tensor(1.0 / self._limit if self.lagrange.get("normalize") else np.ones(self.n_cost), dtype=torch.float32, device=device)

    def dual_step(self, mean_cost: np.ndarray) -> np.ndarray:
        """Dual ascent on the cost limits, scaled by each limit so that one step size fits all three costs."""
        c = self.lagrange
        self._lam_np = np.clip(self._lam_np + float(c["lr"]) * (mean_cost - self._limit) / self._limit, 0.0, float(c.get("max", 100.0)))
        self.lam.copy_(torch.as_tensor(self._lam_np, dtype=torch.float32, device=self.device))
        return self._lam_np.copy()

    def set_budget_floor(self, floor: float) -> None:
        """Budget curriculum: prior draws come from [floor, 1]."""
        self.u_floor.fill_(float(floor))

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return self.rate_min * (self.rate_max / self.rate_min) ** u

    def encode(self, r: torch.Tensor) -> torch.Tensor:
        return (torch.minimum(torch.maximum(r, self.rate_min), self.rate_max) / self.rate_min).log() / (self.rate_max / self.rate_min).log()

    def add(self, rows: torch.Tensor, ended: np.ndarray, costs: np.ndarray | None = None) -> None:
        """`costs` is the same data as the row's cost columns; passing it avoids a GPU read."""
        self._cum_row.copy_(torch.as_tensor(self._cum_np, device=self.device))
        self.cum[self.pos * self.n_envs + self.env_index] = self._cum_row
        if costs is None:
            costs = rows[:, self.i_cost:self.i_cost + self.n_cost].cpu().numpy()
        self._cum_np += costs.astype(np.float64)
        self._cum_np[ended] = 0.0
        super().add(rows, ended)

    def levels(self, row_start, env, t, length, rate, capacity):
        """Deficit at step t for new rates: (V, self.n_cost). Rows past an episode's end are never reached, because cummin looks backward."""
        rows = (row_start[:, None] + self.steps_L) % self.rows
        idx = rows * self.n_envs + env[:, None]
        S = self.cum[idx]  # (V, L, 3)
        F = S - rate[:, None, :] * self.steps_f[None, :, None]
        low = torch.cummin(F, dim=1).values
        i = t[:, None, None].expand(-1, 1, self.n_cost)
        return (F.gather(1, i) - low.gather(1, i)).squeeze(1)

    def window_levels(self, row_start, env, t, rate, cost_t):
        """Deficits at t and t + 1 for new rates, looking back at most W steps: (V, self.n_cost) each. Steps before the episode's start repeat step 0."""
        k = (t[:, None] - self.steps_back).clamp_min(0)  # (V, W + 1), newest first
        idx = ((row_start[:, None] + k) % self.rows) * self.n_envs + env[:, None]
        F = self.cum[idx] - rate[:, None, :] * k[:, :, None].to(torch.float64)
        F_next = F[:, 0] + cost_t - rate
        return F[:, 0] - F.amin(dim=1), F_next - torch.minimum(F[:, :-1].amin(dim=1), F_next)

    def sample(self):
        d, B, V, E, i = self.device, self.batch_size, self.nb_virtual, self.n_envs, self.i_rew
        pick = self.valid[(torch.rand(B, device=d) * self.n_valid).long()]
        b = self.store[pick]
        pv = pick[:V]
        env = pv % E
        row = torch.div(pv, E, rounding_mode="floor")
        start, length = self.ep_start[pv], self.ep_len[pv]
        cur = (row - start) % self.rows
        fut = cur + (torch.rand(V, device=d) * (length - cur).float()).long()
        fut_idx = ((fut + start) % self.rows) * E + env
        goal_v = self.store[fut_idx][:, self.i_next + self.a0:self.i_next + self.a1]
        if self.keep_cmd is not None:
            goal_v = torch.where(self.keep_cmd, b[:V, self.d0:self.d0 + 3], goal_v)

        old_u = b[:V, self.d0 + 3:self.d1].to(torch.float64)
        spent = (self.cum[fut_idx] + self.store[fut_idx][:, self.i_cost:self.i_cost + self.n_cost].to(torch.float64)) / (fut + 1).to(torch.float64)[:, None]
        achieved_u = self.encode(spent)
        prior_u = self.u_floor + (1.0 - self.u_floor) * torch.rand((V, self.n_cost), device=d, dtype=torch.float64)
        if self.p_corner:
            hi = torch.maximum(self.u_floor, torch.full_like(self.u_floor, 0.25))
            tight = self.u_floor + (hi - self.u_floor) * torch.rand(V, device=d, dtype=torch.float64)
            corner = torch.ones((V, self.n_cost), device=d, dtype=torch.float64)
            corner.scatter_(1, torch.randint(0, self.n_cost, (V, 1), device=d), tight[:, None])
            prior_u = torch.where((torch.rand(V, device=d) < self.p_corner)[:, None], corner, prior_u)
        choice = torch.rand((V, self.n_cost), device=d)
        u = torch.where(choice < self.p_keep, old_u, torch.where(choice < self.p_keep + self.p_achieved, achieved_u, prior_u))

        rate = self.decode(u)
        cap = torch.maximum(rate * self.span, self.cap_floor)
        cost_t = b[:V, self.i_cost:self.i_cost + self.n_cost].to(torch.float64)
        if self.episode_budget:
            deficit = self.cum[pv]
            next_deficit = deficit + cost_t
        elif self.window:
            deficit, next_deficit = self.window_levels(start, env, cur, rate, cost_t)
        else:
            deficit = self.levels(start, env, cur, length, rate, cap)
            next_deficit = (deficit + cost_t - rate).clamp_min(0.0)
        level = (1.0 - deficit / cap).clamp(-1.0, 1.0).float()
        next_level = (1.0 - next_deficit / cap).clamp(-1.0, 1.0).float()
        overdrawn = (next_deficit > cap).any(dim=1)

        goal = torch.cat((goal_v, u.float()), dim=1)
        b[:V, self.d0:self.d1] = goal
        b[:V, self.i_next + self.d0:self.i_next + self.d1] = goal
        b[:V, self.i_level:self.obs_dim] = level
        b[:V, self.i_next + self.i_level:self.i_next + self.obs_dim] = next_level
        base = relabel_reward(self.cfg, b[:V, self.i_next + self.a0:self.i_next + self.a1], goal_v, b[:V, i + 2], b[:V, i + 3], b[:V, i + 4], b[:V, i + 5], b[:V, i + 6])
        if self.soft_overdraft:
            factor = torch.exp(-((next_deficit - cap).clamp_min(0.0) / cap).sum(dim=1)).float()
            alive = torch.minimum(base, base * factor) + self.alive_bonus
        else:
            alive = torch.where(overdrawn, torch.zeros_like(base), base) + self.alive_bonus
        b[:V, i] = torch.where(b[:V, i + 6] > 0.5, base, alive)
        if self.lagrange:
            b[:, i] = b[:, i] - (self.lam * self.lag_scale * b[:, self.i_cost:self.i_cost + self.n_cost]).sum(1)
        return self._finish(b[:, :self.obs_dim], b[:, self.i_act:self.i_rew], b[:, i], b[:, self.i_next:self.i_act], b[:, i + 1], self.gamma_pow[1], b)
