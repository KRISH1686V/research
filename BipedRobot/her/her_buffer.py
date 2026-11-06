"""
Hindsight Experience Replay (HER) Buffer for Goal-Conditioned Locomotion.

Features:
- Episode-based storage with fast transition extraction.
- Relabeling strategies: 'future' (default k=4), 'final', 'episode', 'random'.
- Dynamic reward recomputation using AdaptiveRewardFunction with current lambda weights.
- Collection of relabeled hindsight violations for dual-ascent automatic reward correction.
- Adaptive Goal Curriculum that expands target velocity envelope upon meeting tracking criteria.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class HERReplayBuffer:
    """
    Episode-based Replay Buffer with Hindsight Goal Relabeling and Reward Recomputation.
    """

    def __init__(
        self,
        capacity_transitions: int = 200_000,
        relabel_strategy: str = "future",
        relabel_k: int = 4,
        goal_sampler: Optional[Callable[[], np.ndarray]] = None,
    ):
        assert relabel_strategy in ("future", "final", "episode", "random"), (
            f"Unknown relabel strategy: {relabel_strategy}"
        )
        self.capacity: int = capacity_transitions
        self.relabel_strategy: str = relabel_strategy
        self.relabel_k: int = relabel_k
        self.goal_sampler: Optional[Callable[[], np.ndarray]] = goal_sampler

        # List of episodes, where each episode is a list of transition dicts
        self.episodes: List[List[Dict[str, Any]]] = []
        self.total_transitions: int = 0
        self._current_episode: List[Dict[str, Any]] = []

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        next_achieved_goal: np.ndarray,
        done: bool,
        truncated: bool,
        violations: Dict[str, float],
    ) -> None:
        """Add a single step to the ongoing trajectory."""
        # Extract scalar violation metrics
        scalar_violations = {}
        for k, v in violations.items():
            if np.isscalar(v):
                scalar_violations[k] = float(v)
            elif isinstance(v, (np.ndarray, list)) and len(v) == 1:
                scalar_violations[k] = float(v[0])

        transition = {
            "obs": np.asarray(obs, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "achieved_goal": np.asarray(achieved_goal, dtype=np.float32),
            "desired_goal": np.asarray(desired_goal, dtype=np.float32),
            "next_achieved_goal": np.asarray(next_achieved_goal, dtype=np.float32),
            "done": bool(done),
            "truncated": bool(truncated),
            "violations": scalar_violations,
        }
        self._current_episode.append(transition)

        if done or truncated:
            self.finalize_episode()

    def finalize_episode(self) -> None:
        """Push the completed trajectory to buffer storage and enforce capacity limit."""
        if len(self._current_episode) == 0:
            return
        
        ep_len = len(self._current_episode)
        self.episodes.append(self._current_episode)
        self.total_transitions += ep_len
        self._current_episode = []

        # Evict oldest episodes if over capacity
        while self.total_transitions > self.capacity and len(self.episodes) > 1:
            evicted = self.episodes.pop(0)
            self.total_transitions -= len(evicted)

    def __len__(self) -> int:
        return self.total_transitions

    def sample(
        self,
        batch_size: int,
        reward_fn: Optional[Any] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[float]]]:
        """
        Sample a batch of transitions with HER relabeling.
        
        Returns:
            batch_tensors: Dict containing torch Tensors for actor and critic:
              'obs', 'actions', 'rewards', 'next_obs', 'dones', 'goals'
            hindsight_violations: Dict of lists containing constraint violation signals
              on the relabeled transitions (used for dual-ascent multiplier update).
        """
        if len(self.episodes) == 0:
            raise ValueError("Cannot sample from empty replay buffer.")

        # Uniformly sample episodes weighted by length
        ep_indices = random.choices(range(len(self.episodes)), weights=[len(ep) for ep in self.episodes], k=batch_size)

        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        batch_goals = []
        
        hindsight_violations: Dict[str, List[float]] = {}

        # HER relabeling probability: k / (k + 1)
        relabel_prob = self.relabel_k / (self.relabel_k + 1.0) if self.relabel_k > 0 else 0.0

        for ep_idx in ep_indices:
            episode = self.episodes[ep_idx]
            ep_len = len(episode)
            t = random.randint(0, ep_len - 1)
            transition = episode[t]

            should_relabel = (random.random() < relabel_prob)
            
            if should_relabel:
                # Select hindsight goal based on strategy
                if self.relabel_strategy == "future":
                    # Sample a future step in the same episode
                    future_t = random.randint(t, ep_len - 1)
                    relabeled_goal = episode[future_t]["achieved_goal"].copy()
                elif self.relabel_strategy == "final":
                    relabeled_goal = episode[-1]["achieved_goal"].copy()
                elif self.relabel_strategy == "episode":
                    any_t = random.randint(0, ep_len - 1)
                    relabeled_goal = episode[any_t]["achieved_goal"].copy()
                elif self.relabel_strategy == "random" and self.goal_sampler is not None:
                    relabeled_goal = self.goal_sampler()
                else:
                    relabeled_goal = episode[-1]["achieved_goal"].copy()

                goal = relabeled_goal
                
                # Recompute reward using adaptive reward function with current lambdas
                if reward_fn is not None:
                    r, _ = reward_fn.compute_reward(
                        achieved_goal=transition["achieved_goal"],
                        desired_goal=goal,
                        violations=transition["violations"],
                    )
                    reward = float(r)
                else:
                    reward = transition["reward"]

                # Accumulate violation signals on relabeled transition
                for k, v in transition["violations"].items():
                    if k not in hindsight_violations:
                        hindsight_violations[k] = []
                    hindsight_violations[k].append(float(v))
            else:
                goal = transition["desired_goal"].copy()
                if reward_fn is not None:
                    r, _ = reward_fn.compute_reward(
                        achieved_goal=transition["achieved_goal"],
                        desired_goal=goal,
                        violations=transition["violations"],
                    )
                    reward = float(r)
                else:
                    reward = transition["reward"]

            # Goal-conditioned states: concatenate proprioceptive state and target goal
            s = np.concatenate([transition["obs"], goal], axis=0)
            s_next = np.concatenate([transition["next_obs"], goal], axis=0)

            batch_obs.append(s)
            batch_actions.append(transition["action"])
            batch_rewards.append(reward)
            batch_next_obs.append(s_next)
            batch_dones.append(1.0 if transition["done"] else 0.0)
            batch_goals.append(goal)

        # Convert to PyTorch tensors
        batch_tensors = {
            "obs": torch.as_tensor(np.array(batch_obs, dtype=np.float32)),
            "actions": torch.as_tensor(np.array(batch_actions, dtype=np.float32)),
            "rewards": torch.as_tensor(np.array(batch_rewards, dtype=np.float32)).unsqueeze(-1),
            "next_obs": torch.as_tensor(np.array(batch_next_obs, dtype=np.float32)),
            "dones": torch.as_tensor(np.array(batch_dones, dtype=np.float32)).unsqueeze(-1),
            "goals": torch.as_tensor(np.array(batch_goals, dtype=np.float32)),
        }

        return batch_tensors, hindsight_violations


class GoalCurriculum:
    """
    Curriculum learning for expanding target velocity distribution.
    
    Expands velocity bounds when rolling success rate exceeds the threshold.
    """

    def __init__(
        self,
        initial_vx_range: Tuple[float, float] = (-0.1, 0.2),
        initial_vy_range: Tuple[float, float] = (-0.05, 0.05),
        initial_wz_range: Tuple[float, float] = (-0.15, 0.15),
        max_vx_range: Tuple[float, float] = (-0.3, 0.6),
        max_vy_range: Tuple[float, float] = (-0.25, 0.25),
        max_wz_range: Tuple[float, float] = (-0.5, 0.5),
        success_threshold: float = 0.70,
        expansion_rate: float = 1.15,
        window_size: int = 50,
    ):
        self.curr_vx = list(initial_vx_range)
        self.curr_vy = list(initial_vy_range)
        self.curr_wz = list(initial_wz_range)

        self.max_vx = list(max_vx_range)
        self.max_vy = list(max_vy_range)
        self.max_wz = list(max_wz_range)

        self.success_threshold = success_threshold
        self.expansion_rate = expansion_rate
        self.window_size = window_size
        self.recent_outcomes: List[bool] = []

    def sample_goal(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample target goal from current curriculum envelope."""
        np_rng = rng or np.random.default_rng()
        vx = np_rng.uniform(self.curr_vx[0], self.curr_vx[1])
        vy = np_rng.uniform(self.curr_vy[0], self.curr_vy[1])
        wz = np_rng.uniform(self.curr_wz[0], self.curr_wz[1])
        return np.array([vx, vy, wz], dtype=np.float32)

    def record_episode(self, success: bool) -> bool:
        """Record episode outcome and expand bounds if criteria is met."""
        self.recent_outcomes.append(bool(success))
        if len(self.recent_outcomes) > self.window_size:
            self.recent_outcomes.pop(0)

        expanded = False
        if len(self.recent_outcomes) >= self.window_size:
            sr = sum(self.recent_outcomes) / len(self.recent_outcomes)
            if sr >= self.success_threshold:
                expanded = self._expand()
                self.recent_outcomes.clear()
        return expanded

    def _expand(self) -> bool:
        """Expand velocity envelope towards max ranges."""
        expanded = False
        for i in range(2):
            if abs(self.curr_vx[i]) < abs(self.max_vx[i]):
                self.curr_vx[i] = np.clip(self.curr_vx[i] * self.expansion_rate, self.max_vx[0], self.max_vx[1])
                expanded = True
            if abs(self.curr_vy[i]) < abs(self.max_vy[i]):
                self.curr_vy[i] = np.clip(self.curr_vy[i] * self.expansion_rate, self.max_vy[0], self.max_vy[1])
                expanded = True
            if abs(self.curr_wz[i]) < abs(self.max_wz[i]):
                self.curr_wz[i] = np.clip(self.curr_wz[i] * self.expansion_rate, self.max_wz[0], self.max_wz[1])
                expanded = True
        return expanded

    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "vx": (float(self.curr_vx[0]), float(self.curr_vx[1])),
            "vy": (float(self.curr_vy[0]), float(self.curr_vy[1])),
            "wz": (float(self.curr_wz[0]), float(self.curr_wz[1])),
        }
