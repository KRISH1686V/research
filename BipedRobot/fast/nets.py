"""Actor and twin critic with the same layers, init and parameter names as SB3's MultiInputPolicy for SAC.

Inputs are the concatenated dict observation in SB3's order: achieved_goal, desired_goal, observation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
TANH_EPS = 1e-6
_HALF_LOG_2PI = 0.9189385332046727


def _mlp(dims: list[int], out: bool, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if out and i == len(dims) - 2:
            break
        if layer_norm:
            layers.append(nn.LayerNorm(b))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def actor_inputs(obs_dim: int, n_goal: int, n_levels: int, hide: set[str]) -> list[int]:
    """Indices of the concatenated observation the policy may read. `hide` can name "achieved" and "levels", which a real robot cannot measure."""
    keep = range(3 if "achieved" in hide else 0, obs_dim - (n_levels if "levels" in hide else 0))
    assert 3 + n_goal + n_levels < obs_dim
    return list(keep)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, init_log_std: float | None = None, keep: list[int] | torch.Tensor | None = None, mirror: dict | None = None):
        """mirror: {obs_perm, obs_sign, act_perm} makes the policy exactly left-right symmetric, the designer's equivariant actor: pi(s) = 1/2 (f(s) + M f(M s)). Off by default."""
        super().__init__()
        self.register_buffer("keep", None if keep is None else torch.as_tensor(keep, dtype=torch.int64))
        self.register_buffer("obs_perm", None if mirror is None else torch.as_tensor(mirror["obs_perm"], dtype=torch.int64))
        self.register_buffer("obs_sign", None if mirror is None else torch.as_tensor(mirror["obs_sign"], dtype=torch.float32))
        self.register_buffer("act_perm", None if mirror is None else torch.as_tensor(mirror["act_perm"], dtype=torch.int64))
        self.latent_pi = _mlp([obs_dim if keep is None else len(keep), hidden, hidden], out=False)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        if init_log_std is not None:
            with torch.no_grad():
                self.log_std.weight.mul_(0.01)
                self.log_std.bias.fill_(init_log_std)

    def _raw(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_pi(obs if self.keep is None else obs[..., self.keep])
        return self.mu(h), self.log_std(h)

    def dist_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._raw(obs)
        if self.obs_perm is not None:
            mean_m, log_std_m = self._raw(obs[..., self.obs_perm] * self.obs_sign)
            mean, log_std = 0.5 * (mean + mean_m[..., self.act_perm]), 0.5 * (log_std + log_std_m[..., self.act_perm])
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised squashed sample and its log-probability, as SquashedDiagGaussianDistribution."""
        mean, log_std = self.dist_params(obs)
        noise = torch.randn_like(mean)
        gaussian = mean + log_std.exp() * noise
        action = gaussian.tanh()
        log_prob = (-0.5 * noise * noise - log_std - _HALF_LOG_2PI).sum(-1)
        log_prob = log_prob - (1.0 - action * action + TANH_EPS).log().sum(-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.dist_params(obs)[0].tanh()


def actor_from_checkpoint(state: dict) -> Actor:
    """Rebuild the policy from a plain-weights checkpoint written by train.py."""
    sd = state["actor"]
    mirror = {k: sd[k] for k in ("obs_perm", "obs_sign", "act_perm")} if "obs_perm" in sd else None
    actor = Actor(state["obs_dim"], state["act_dim"], keep=sd.get("keep"), mirror=mirror)
    actor.load_state_dict(sd)
    return actor.eval()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, layer_norm: bool = False):
        super().__init__()
        self.qf0 = _mlp([obs_dim + act_dim, hidden, hidden, 1], out=True, layer_norm=layer_norm)
        self.qf1 = _mlp([obs_dim + act_dim, hidden, hidden, 1], out=True, layer_norm=layer_norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((obs, act), dim=-1)
        return self.qf0(x).squeeze(-1), self.qf1(x).squeeze(-1)
