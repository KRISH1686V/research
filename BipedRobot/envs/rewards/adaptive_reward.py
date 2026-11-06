"""
Adaptive Reward Function with Dual-Ascent Automatic Correction for Biped Locomotion.

Formulation:
  r(s, a, g) = r_track(achieved_goal, g) - sum_k lambda_k * c_k(s, a)
  
Automatic Reward Correction:
  Treats each regularizer/penalty c_k as a constraint with target budget d_k.
  Dual ascent update performed on hindsight (relabeled) transitions:
    lambda_k <- clip(lambda_k + eta_k * (mean_violation_k_HER - d_k), lambda_min_k, lambda_max_k)
  with exponential moving average (EMA) smoothing over rollout batches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Standard constraint terms and their default parameters
DEFAULT_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "torso_tilt": {
        "budget": 0.05,
        "fixed_weight": 2.0,
        "lr": 0.05,
        "lambda_min": 0.0,
        "lambda_max": 10.0,
    },
    "torque_saturation_ratio": {
        "budget": 0.05,
        "fixed_weight": 1.5,
        "lr": 0.05,
        "lambda_min": 0.0,
        "lambda_max": 10.0,
    },
    "foot_slip": {
        "budget": 0.05,
        "fixed_weight": 0.8,
        "lr": 0.02,
        "lambda_min": 0.0,
        "lambda_max": 8.0,
    },
    "action_rate": {
        "budget": 0.02,
        "fixed_weight": 0.3,
        "lr": 0.01,
        "lambda_min": 0.0,
        "lambda_max": 5.0,
    },
    "joint_limit_proximity": {
        "budget": 0.01,
        "fixed_weight": 1.5,
        "lr": 0.05,
        "lambda_min": 0.0,
        "lambda_max": 10.0,
    },
    "foot_clearance": {
        "budget": 0.02,
        "fixed_weight": 0.6,
        "lr": 0.02,
        "lambda_min": 0.0,
        "lambda_max": 6.0,
    },
    "energy": {
        "budget": 0.08,
        "fixed_weight": 0.15,
        "lr": 0.01,
        "lambda_min": 0.0,
        "lambda_max": 4.0,
    },
    "gait_asymmetry": {
        "budget": 0.10,
        "fixed_weight": 0.5,
        "lr": 0.02,
        "lambda_min": 0.0,
        "lambda_max": 5.0,
    },
}


class AdaptiveRewardFunction:
    """
    Reward computer with dual-ascent automatic weight correction.
    
    Supports two modes:
      - 'adaptive': dynamic dual ascent updates using relabeled hindsight transitions.
      - 'fixed': static hand-tuned weights (baseline reproduction).
    """

    def __init__(
        self,
        reward_mode: str = "adaptive",
        config: Optional[Dict[str, Any]] = None,
        sigma_v: float = 0.35,
        sigma_w: float = 0.35,
        ema_beta: float = 0.85,
    ):
        assert reward_mode in ("adaptive", "fixed"), f"Unknown reward mode: {reward_mode}"
        self.reward_mode = reward_mode
        self.config = config or {}
        self.sigma_v = float(self.config.get("sigma_v", sigma_v))
        self.sigma_w = float(self.config.get("sigma_w", sigma_w))
        self.ema_beta = float(self.config.get("ema_beta", ema_beta))

        # Setup constraint terms
        custom_constraints = self.config.get("constraints", {})
        self.constraint_names: List[str] = list(DEFAULT_CONSTRAINTS.keys())
        
        self.budgets: Dict[str, float] = {}
        self.lambdas: Dict[str, float] = {}
        self.fixed_weights: Dict[str, float] = {}
        self.lrs: Dict[str, float] = {}
        self.lambda_mins: Dict[str, float] = {}
        self.lambda_maxs: Dict[str, float] = {}
        self.ema_violations: Dict[str, float] = {}

        for k, default_vals in DEFAULT_CONSTRAINTS.items():
            overrides = custom_constraints.get(k, {})
            self.budgets[k] = float(overrides.get("budget", default_vals["budget"]))
            init_w = float(overrides.get("fixed_weight", default_vals["fixed_weight"]))
            self.fixed_weights[k] = init_w
            self.lambdas[k] = init_w  # Initialize multiplier to nominal weight
            self.lrs[k] = float(overrides.get("lr", default_vals["lr"]))
            self.lambda_mins[k] = float(overrides.get("lambda_min", default_vals["lambda_min"]))
            self.lambda_maxs[k] = float(overrides.get("lambda_max", default_vals["lambda_max"]))
            self.ema_violations[k] = self.budgets[k]  # start at budget

        # Tracking history for analysis and plotting
        self.lambda_history: Dict[str, List[float]] = {k: [self.lambdas[k]] for k in self.constraint_names}
        self.violation_history: Dict[str, List[float]] = {k: [] for k in self.constraint_names}
        self.update_count: int = 0

    def compute_tracking_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> float:
        """
        Compute Gaussian kernel velocity tracking reward:
          r_track = exp(- ||v_xy - v_xy*||^2 / (2 * sigma_v^2)) * exp(- (w_z - w_z*)^2 / (2 * sigma_w^2))
        """
        v_diff = achieved_goal[:2] - desired_goal[:2]
        w_diff = achieved_goal[2] - desired_goal[2]
        v_err_sq = float(np.sum(v_diff ** 2))
        w_err_sq = float(w_diff ** 2)

        r_v = np.exp(- v_err_sq / (2.0 * (self.sigma_v ** 2)))
        r_w = np.exp(- w_err_sq / (2.0 * (self.sigma_w ** 2)))
        return float(r_v * r_w)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        violations: Dict[str, Union[float, np.ndarray]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total scalar reward and component breakdown:
          r = r_track - sum_k lambda_k * c_k
        """
        r_track = self.compute_tracking_reward(achieved_goal, desired_goal)
        
        penalty_sum = 0.0
        components: Dict[str, float] = {"r_track": r_track}

        active_weights = self.lambdas if self.reward_mode == "adaptive" else self.fixed_weights

        for k in self.constraint_names:
            v_val = float(violations.get(k, 0.0))
            w = active_weights[k]
            pen = w * v_val
            penalty_sum += pen
            components[f"penalty_{k}"] = pen
            components[f"violation_{k}"] = v_val
            components[f"lambda_{k}"] = w

        total_reward = r_track - penalty_sum
        components["total_reward"] = total_reward
        components["penalty_sum"] = penalty_sum

        return total_reward, components

    def update_multipliers(
        self,
        hindsight_violations: Dict[str, Union[List[float], np.ndarray, float]],
    ) -> Dict[str, float]:
        """
        Update dual multipliers lambda_k using violations evaluated on hindsight transitions:
          lambda_k <- clip(lambda_k + eta_k * (EMA(violation_k) - d_k), lambda_min, lambda_max)
          
        If reward_mode == 'fixed', multipliers remain pinned at fixed_weight.
        """
        self.update_count += 1
        metrics: Dict[str, float] = {}

        if self.reward_mode == "fixed":
            # Fixed ablation baseline: do not adjust weights
            for k in self.constraint_names:
                v = hindsight_violations.get(k, 0.0)
                mean_v = float(np.mean(v)) if hasattr(v, "__len__") else float(v)
                metrics[f"lambda/{k}"] = self.fixed_weights[k]
                metrics[f"violation_mean/{k}"] = mean_v
                metrics[f"budget/{k}"] = self.budgets[k]
            return metrics

        # Adaptive mode: dual ascent on hindsight violations
        for k in self.constraint_names:
            v = hindsight_violations.get(k, 0.0)
            mean_v = float(np.mean(v)) if hasattr(v, "__len__") else float(v)
            
            # EMA smoothing of empirical violation rate
            self.ema_violations[k] = self.ema_beta * self.ema_violations[k] + (1.0 - self.ema_beta) * mean_v
            effective_v = self.ema_violations[k]

            budget = self.budgets[k]
            lr = self.lrs[k]
            grad = effective_v - budget

            # Dual ascent step
            new_lambda = self.lambdas[k] + lr * grad
            new_lambda = float(np.clip(new_lambda, self.lambda_mins[k], self.lambda_maxs[k]))
            self.lambdas[k] = new_lambda

            # Record history
            self.lambda_history[k].append(new_lambda)
            self.violation_history[k].append(mean_v)

            # Log metrics
            metrics[f"lambda/{k}"] = new_lambda
            metrics[f"violation_mean/{k}"] = mean_v
            metrics[f"violation_ema/{k}"] = effective_v
            metrics[f"budget/{k}"] = budget
            metrics[f"dual_gradient/{k}"] = grad

        return metrics

    def get_lambdas(self) -> Dict[str, float]:
        """Return current multipliers (or fixed weights if in fixed mode)."""
        if self.reward_mode == "fixed":
            return dict(self.fixed_weights)
        return dict(self.lambdas)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "reward_mode": self.reward_mode,
            "lambdas": dict(self.lambdas),
            "ema_violations": dict(self.ema_violations),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.reward_mode = state.get("reward_mode", self.reward_mode)
        if "lambdas" in state:
            self.lambdas.update(state["lambdas"])
        if "ema_violations" in state:
            self.ema_violations.update(state["ema_violations"])
        self.update_count = state.get("update_count", self.update_count)
