"""Token-bucket gait budgets as part of the goal.

PROVISIONAL. Built from the Q1 roadmap and the standard token bucket, because the train-once-plan doc was not available. The reward
gate, budget ranges and burst length below are my choices and must be checked against that doc if it turns up.

Goal: [vx, vy, wz, u_flight, u_slip, u_torque]. Each u in [0, 1] maps on a log scale to a rate r, the cost allowed per control step.
Costs are measured as evaluation_suite_v2 measures them (see gait_costs.py): flight fraction of substeps, stance slip in m/s, and
fraction of actuators at the torque limit.

Bucket, per cost, starting full with capacity B = r * burst_steps:
    b(t+1) = min(B, b(t) + r - c(t))        with no floor, so overdraft shows as b < 0
Equivalently, with deficit d = B - b:  d(t+1) = max(0, d(t) + c(t) - r),  d(0) = 0.
While every bucket is non-negative, total cost in any window of T steps is at most r * T + B.

Reward: the base reward (tracking + progress + smoothing) applies only while no bucket is overdrawn, otherwise 0. A fall is still
the fall penalty. This follows the budget-augmented reward of Saute RL. The policy also sees each bucket's level b / B, clipped to [-1, 1].

For survival, token_budget.alive_bonus adds a constant to every step that does not end in a fall, so an overdrawn bucket does not remove the
reason to stay upright. token_budget.curriculum draws budgets from [1 - d, 1] with difficulty d, so they start loose and tighten as d rises.

token_budget.window_steps W bounds the bucket's memory, so an overdraft can be recovered from:
    F(k) = S(k) - r * k,   d(t) = F(t) - min[max(0, t - W) <= k <= t] F(k)
d(t) <= B for all t means cost in any window of T <= W steps is at most r * T + B, the same guarantee for windows up to W. Without it (the default),
a tight budget stays overdrawn for the rest of the episode after one early mistake, and the base reward is gone whatever the policy does next.

token_budget.mode "episode" replaces the bucket with one allowance per episode, the comparison the roadmap names: overdrawn once S(t) > r * max_episode_steps. A burst can spend
the whole allowance, which is what the bucket exists to prevent.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from BipedRobot.fast.gait_costs import GaitCostEnv
from BipedRobot.fast.valid_env import ValidLocoEnv

COSTS = ("flight", "slip", "torque")
ALL_COSTS = ("flight", "slip", "torque", "steps", "lowsteps", "tiltstance", "partsole", "shake", "jshake")
# token_budget.costs picks from these. "steps" counts foot touchdowns per control step. "lowsteps" counts touchdowns whose swing never lifted the foot's
# lowest mesh point token_budget.low_step_mm (default 20) above the floor, measured as evaluation_suite_v2 measures clearance, once per control step.
# "tiltstance" counts feet on the floor whose sole is tilted more than token_budget.tilt_deg (default 8) from its orientation at reset, where the
# robot stands with flat feet: 0, 1 or 2 per control step.
# "partsole" counts feet on the floor whose sole's highest point is more than token_budget.sole_gap_mm (default 2) above its lowest (ValidLocoEnv.sole_gap,
# from the sole's flat face in the foot mesh): a foot down on its heel, toe, edge or a corner. 0, 1 or 2 per control step. At most six costs at once.
# "shake" (added 10 Nov 2026) is the mean over the joints of the squared second difference of the action, a_t - 2 a_(t-1) + a_(t-2), in action units
# (info["action_second_diff"] from HerLocoEnv). A smooth 1 Hz stride adds little to it; step-to-step jitter, which the robot shows as vibration, adds most.
# The robot can compute it exactly from the targets it sends.
# "jshake" (added 11 Nov 2026) is the same second difference on the measured joint angles (true qpos, rad^2, mean over joints). Under SAC's sampled actions
# "shake" is 4.4-4.6x its deterministic value, so tight shake budgets were never met in training; "jshake" is 1.5-1.6x, because the servos filter the noise.
# token_budget.overdraft "soft" (added 11 Nov 2026; default "hard") scales the task reward by exp(-sum of excess / capacity) while overdrawn, instead of
# zeroing it, so an overdrawn policy is still paid for spending less. Only the positive part is scaled: min(r, r * factor), so overdrawing never pays.
# The alive bonus is kept either way, so falling stays worse than any overdraft. budget_replay.py applies the same rule to relabeled samples.


def decode_rate(u: np.ndarray, rate_min: np.ndarray, rate_max: np.ndarray) -> np.ndarray:
    return rate_min * (rate_max / rate_min) ** u


def encode_rate(r: np.ndarray, rate_min: np.ndarray, rate_max: np.ndarray) -> np.ndarray:
    return np.log(np.clip(r, rate_min, rate_max) / rate_min) / np.log(rate_max / rate_min)


class BudgetLocoEnv(GaitCostEnv, ValidLocoEnv):
    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        self._deficit = np.zeros(3, dtype=np.float64)
        self._u = np.ones(3, dtype=np.float32)
        self._rate = np.ones(3, dtype=np.float64)
        self._capacity = np.ones(3, dtype=np.float64)
        self._levels_ready = False
        super().__init__(config=config, render_mode=render_mode)
        if self.torque_penalty_weight:
            raise ValueError("torque_penalty_weight would apply twice in BudgetLocoEnv; use the torque budget instead")
        b = self.config["token_budget"]
        self.cost_names = tuple(b.get("costs", COSTS))
        if set(self.cost_names) - set(ALL_COSTS) or len(set(self.cost_names)) != len(self.cost_names):
            raise ValueError(f"token_budget.costs must be distinct names from {ALL_COSTS}, got {self.cost_names}")
        self.n_cost = n = len(self.cost_names)
        self.rate_min = np.asarray(b["rate_min"], dtype=np.float64)
        self.rate_max = np.asarray(b["rate_max"], dtype=np.float64)
        if self.rate_min.shape != (n,) or self.rate_max.shape != (n,):
            raise ValueError(f"token_budget.rate_min and rate_max need one value per cost: {self.cost_names}")
        self._deficit, self._u = np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float32)
        self._rate, self._capacity = np.ones(n, dtype=np.float64), np.ones(n, dtype=np.float64)
        self.burst_steps = float(b["burst_steps"])
        self.p_unconstrained = float(b["p_unconstrained"])
        self.slip_clip = float(b.get("slip_clip", 1.0))
        self.alive_bonus = float(b.get("alive_bonus", 0.0))
        self.budget_curriculum = bool(b.get("curriculum", False))
        # token_budget.capacity_floor (12 Nov 2026, default 0): per-cost minimum capacity, B = max(rate x burst, floor). Flight is counted in quarters
        # (4 physics substeps a control step), so below u about 0.23 one 5 ms flight event (0.25) emptied the whole bucket for up to 100 steps; a token
        # bucket's burst must hold at least its largest single unit (as in RFC 2697). token_budget.settle_steps (default 0): costs are not counted on
        # the first control steps after reset, where the feet start exactly at floor height and the first substep reads as flight.
        self.capacity_floor = np.asarray(b.get("capacity_floor", [0.0] * n), dtype=np.float64)
        if self.capacity_floor.shape != (n,) or (self.capacity_floor < 0).any():
            raise ValueError(f"token_budget.capacity_floor needs one value >= 0 per cost: {self.cost_names}")
        self.settle_steps = int(b.get("settle_steps", 0))
        # token_budget.p_corner (13 Nov 2026, default 0): share of budgeted training episodes that tighten ONE cost (u in [floor, max(floor, 0.25)])
        # with the others loose; uniform draws make "one cost tight, the rest loose" rare (about 2-3% of episodes), and flight alone at the tight end
        # was the setting that failed. budget_replay.py draws its prior budgets the same way.
        self.p_corner = float(b.get("p_corner", 0.0))
        if not 0.0 <= self.p_corner <= 1.0:
            raise ValueError("token_budget.p_corner must be in [0, 1]")
        self.u_low = float(b.get("u_low", 0.0))  # 11 Nov 2026: training draws u from [u_low, 1]; below 0 the rate goes under rate_min, so u = 0 sits inside training
        if not -1.0 <= self.u_low <= 0.0:
            raise ValueError("token_budget.u_low must be in [-1, 0]")
        self.window_steps = b.get("window_steps")
        self.episode_budget = b.get("mode", "bucket") == "episode"
        if self.episode_budget and self.window_steps:
            raise ValueError("token_budget: episode mode has no window")
        o = self.observation_space.spaces["observation"].shape[0]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, (o + n,), np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, (3 + n,), np.float32),
        })
        self.low_step_m = float(b.get("low_step_mm", 20.0)) / 1000.0
        self._swing_peak = np.zeros(2, dtype=np.float64)
        self.tilt_rad = float(np.radians(b.get("tilt_deg", 8.0)))
        self.sole_gap_max = float(b.get("sole_gap_mm", 2.0)) / 1000.0
        self.soft_overdraft = b.get("overdraft", "hard") == "soft"
        if b.get("overdraft", "hard") not in ("hard", "soft"):
            raise ValueError("token_budget.overdraft must be hard or soft")
        self._jq = None
        self._sole_normal = None
        self._levels_ready = True

    def _low_steps(self, touchdown: np.ndarray) -> float:
        """Touchdowns this control step whose swing peak stayed below low_step_m. A foot's peak grows while it is off the floor and resets at touchdown. Clearance is ValidLocoEnv.foot_clearance."""
        clear, contact = self.foot_clearance(), self._feet_contact() > 0.5
        low = 0
        for k in range(2):
            if touchdown[k] > 0.5:
                low += int(self._swing_peak[k] < self.low_step_m)
                self._swing_peak[k] = 0.0
            elif not contact[k]:
                self._swing_peak[k] = max(self._swing_peak[k], clear[k])
        return float(low)

    def set_budget(self, u: np.ndarray) -> None:
        self._u = np.asarray(u, dtype=np.float32)
        self._rate = decode_rate(self._u.astype(np.float64), self.rate_min, self.rate_max)
        self._capacity = np.maximum(self._rate * (self.max_episode_steps if self.episode_budget else self.burst_steps), getattr(self, "capacity_floor", 0.0))
        self._deficit = np.zeros(self.n_cost, dtype=np.float64)
        if self.window_steps:
            self._F = np.zeros((int(self.window_steps) + 1, self.n_cost), dtype=np.float64)
            self._t = 0

    def _window_deficit(self, cost: np.ndarray) -> np.ndarray:
        """Next deficit, looking back at most window_steps. self._F is a ring holding F(k) for the steps still in the window."""
        n = len(self._F)
        value = self._F[self._t % n] + cost - self._rate
        self._t += 1
        self._F[self._t % n] = value
        return value - self._F[:min(self._t + 1, n)].min(axis=0)

    def bucket_levels(self) -> np.ndarray:
        """b / B, clipped to [-1, 1]."""
        return np.clip(1.0 - self._deficit / self._capacity, -1.0, 1.0).astype(np.float32)

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs = super()._get_obs_dict()
        if self._levels_ready:
            obs["observation"] = np.concatenate((obs["observation"], self.bucket_levels())).astype(np.float32)
            obs["desired_goal"] = np.concatenate((obs["desired_goal"][:3], self._u)).astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = dict(options or {})
        goal = options.pop("desired_goal", None)
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float32)
            if goal.shape != (3 + self.n_cost,):
                raise ValueError(f"BudgetLocoEnv goals are [vx, vy, wz] then one u per cost: {self.cost_names}")
            options["desired_goal"] = goal[:3]
        # Seed first: base reset reseeds np_random, and the budget is drawn after it.
        self._u = np.ones(self.n_cost, dtype=np.float32)
        self._rate = self.rate_max.copy()
        self._capacity = np.maximum(self._rate * self.burst_steps, getattr(self, "capacity_floor", 0.0))
        self._deficit = np.zeros(self.n_cost, dtype=np.float64)
        self._levels_ready = False
        self._swing_peak = np.zeros(2, dtype=np.float64)
        self._sole_normal = None
        obs, info = super().reset(seed=seed, options=options or None)
        if "jshake" in self.cost_names:
            q0 = self.data.qpos[self.joint_qpos_adrs].astype(np.float64).copy()
            self._jq = [q0, q0.copy()]  # the robot starts still: the first steps cost only their own motion
        if "tiltstance" in self.cost_names:
            self.sole_tilt()  # the reset pose stands on flat feet: this fixes the reference orientation
        if "partsole" in self.cost_names:
            self.sole()  # finds each sole's flat face once, at a reset pose
        if goal is not None:
            self.set_budget(goal[3:])
        elif self.np_random.random() < self.p_unconstrained:
            self.set_budget(np.ones(self.n_cost, dtype=np.float32))
        else:
            floor = 1.0 - self._difficulty * (1.0 - self.u_low) if self.budget_curriculum else self.u_low
            u = self.np_random.uniform(floor, 1.0, self.n_cost)
            if self.p_corner and self.np_random.random() < self.p_corner:
                u = np.ones(self.n_cost)
                u[int(self.np_random.integers(self.n_cost))] = self.np_random.uniform(floor, max(floor, 0.25))
            self.set_budget(u)
        self._levels_ready = True
        obs["observation"] = np.concatenate((obs["observation"], self.bucket_levels())).astype(np.float32)
        obs["desired_goal"] = np.concatenate((obs["desired_goal"][:3], self._u)).astype(np.float32)
        return obs, info

    def step(self, action):
        self._levels_ready = False
        obs, reward, terminated, truncated, info = super().step(action)
        parts = {"flight": info["cost_flight"], "slip": min(info["cost_slip"], self.slip_clip), "torque": info["cost_torque_sat"]}
        if "steps" in self.cost_names:
            parts["steps"] = float(np.sum(info["touchdown_events"]))
        if "lowsteps" in self.cost_names:
            parts["lowsteps"] = self._low_steps(info["touchdown_events"])
        if "tiltstance" in self.cost_names:
            parts["tiltstance"] = float(np.sum((self._feet_contact() > 0.5) & (self.sole_tilt() > self.tilt_rad)))
        if "partsole" in self.cost_names:
            parts["partsole"] = float(np.sum((self._feet_contact() > 0.5) & (self.sole_gap() > self.sole_gap_max)))
        if "shake" in self.cost_names:
            parts["shake"] = float(info["action_second_diff"])
        if "jshake" in self.cost_names:
            q = self.data.qpos[self.joint_qpos_adrs].astype(np.float64).copy()
            parts["jshake"] = float(np.mean((q - 2.0 * self._jq[1] + self._jq[0]) ** 2))
            self._jq = [self._jq[1], q]
        cost = np.array([parts[c] for c in self.cost_names], dtype=np.float32).astype(np.float64)
        if self._step_count <= self.settle_steps:
            cost[:] = 0.0
        if self.episode_budget:
            self._deficit = self._deficit + cost
        else:
            self._deficit = self._window_deficit(cost) if self.window_steps else np.maximum(0.0, self._deficit + cost - self._rate)
        overdrawn = bool((self._deficit > self._capacity).any())
        self._levels_ready = True
        obs["observation"] = np.concatenate((obs["observation"], self.bucket_levels())).astype(np.float32)
        obs["desired_goal"] = np.concatenate((obs["desired_goal"][:3], self._u)).astype(np.float32)
        info["budget_overdrawn"] = overdrawn
        info["budget_cost"] = cost.astype(np.float32)
        if not info["fell"]:
            if self.soft_overdraft:
                factor = float(np.exp(-(np.maximum(0.0, self._deficit - self._capacity) / self._capacity).sum()))
                task = min(reward, reward * factor)
            else:
                task = 0.0 if overdrawn else reward
            reward = task + self.alive_bonus
        return obs, float(reward), terminated, truncated, info
