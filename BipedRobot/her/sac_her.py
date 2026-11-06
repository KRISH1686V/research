"""
Goal-Conditioned Soft Actor-Critic (SAC) implementation tailored for HER-Loco.

Features:
- Goal-conditioned actor and twin critic networks (obs + goal concatenation).
- Reparameterized Squashed Gaussian policy with numerically stable log-prob correction.
- Polyak soft target updates and automatic temperature (alpha) tuning.
- GPU acceleration with CPU fallback.
- Checkpoint serialization.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN: float = -20.0
LOG_STD_MAX: float = 2.0


class Actor(nn.Module):
    """Goal-conditioned Gaussian policy with tanh output squashing."""

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = obs_dim + goal_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        
        # Enforcing Action Bound formula from SAC paper:
        # log_prob = log_prob(u) - sum(log(1 - tanh(u)^2 + eps))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class TwinCritic(nn.Module):
    """Twin Q-networks for clipped double Q-learning."""

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = obs_dim + goal_dim + action_dim

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2


class SACHERAgent:
    """
    Goal-Conditioned SAC Agent integrating HER replay buffer.
    """

    def __init__(
        self,
        obs_dim: int = 42,
        goal_dim: int = 3,
        action_dim: int = 12,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(obs_dim, goal_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinCritic(obs_dim, goal_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinCritic(obs_dim, goal_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatic temperature tuning
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action given proprioceptive state and goal."""
        with torch.no_grad():
            s = np.concatenate([obs, goal], axis=0)
            state_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, mean_action = self.actor.sample(state_t)
            act = mean_action if deterministic else action
            return act.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one gradient update step for actor, critic, and entropy temperature."""
        states = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        # ------------------- 1. Critic Update ------------------- #
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_val = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_val) + F.mse_loss(current_q2, target_val)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ------------------- 2. Actor Update ------------------- #
        sampled_actions, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ------------------- 3. Temperature Update ------------- #
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ------------------- 4. Polyak Target Update ----------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "loss/critic": float(critic_loss.item()),
            "loss/actor": float(actor_loss.item()),
            "loss/alpha": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "entropy": float(-log_probs.mean().item()),
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
