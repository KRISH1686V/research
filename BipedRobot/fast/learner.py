"""SAC update in SB3's order: actor sample, alpha, target, critic, actor, polyak. Modes differ only in how it runs on the GPU.

eager:  plain PyTorch with SB3's Adam settings.
fused:  eager with fused Adam.
graph:  sampling, relabelling and the whole update captured in one CUDA graph, with fused capturable Adam.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .nets import Actor, Critic
from .replay import GpuHerReplay


class SacLearner:
    def __init__(self, replay: GpuHerReplay, obs_dim: int, act_dim: int, device: torch.device, mode: str = "eager",
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, hidden: int = 256,
                 layer_norm: bool = False, avg_q: bool = False, init_log_std: float | None = None, init_alpha: float = 1.0,
                 priv_dim: int = 0, ema_tau: float = 0.0, actor_keep: list[int] | None = None,
                 caps_t: float = 0.0, caps_s: float = 0.0, caps_sigma: float = 0.01):
        if mode not in ("eager", "fused", "graph"):
            raise ValueError(f"unknown mode {mode}")
        self.caps_t, self.caps_s, self.caps_sigma = float(caps_t), float(caps_s), float(caps_sigma)
        self.replay, self.device, self.mode = replay, device, mode
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -float(act_dim)
        self.avg_q, self.layer_norm = avg_q, layer_norm
        self.actor = Actor(obs_dim, act_dim, hidden, init_log_std=init_log_std, keep=actor_keep).to(device)
        self.critic = Critic(obs_dim + priv_dim, act_dim, hidden, layer_norm=layer_norm).to(device)
        self.ema_tau = float(ema_tau)
        self.actor_ema = copy.deepcopy(self.actor) if ema_tau else None
        if self.actor_ema is not None:
            self.actor_ema.requires_grad_(False)
            self._a = list(self.actor.parameters())
            self._e = list(self.actor_ema.parameters())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.log_alpha = torch.full((1,), float(torch.tensor(init_alpha).log()), device=device, requires_grad=True)
        kw = {} if mode == "eager" else {"fused": True}
        if mode == "graph":
            kw["capturable"] = True
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, **kw)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, **kw)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, **kw)
        self._p = [p for p in self.critic.parameters()]
        self._t = [p for p in self.critic_target.parameters()]
        self.stats = torch.zeros(5, device=device)
        self.n_updates = 0
        self._graph = None
        self._act_graph = None
        self._act_in = None
        self._act_out = None

    def _combine(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q0 + q1) if self.avg_q else torch.min(q0, q1)

    def _smoothness(self, obs: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Policy smoothness as a training loss, not a reward (CAPS, Mysore et al. 2021), on the mean action the robot would run: lambda_T |mu(s) - mu(s')|^2 + lambda_S |mu(s) - mu(s + noise)|^2.
        Squared, unlike the paper's plain distance. The temporal term skips steps that end the episode."""
        mean = self.actor.deterministic(obs)
        loss = torch.zeros((), device=self.device)
        if self.caps_t:
            loss = loss + self.caps_t * ((1.0 - done) * (mean - self.actor.deterministic(next_obs)).pow(2).sum(-1)).mean()
        if self.caps_s:
            noisy = obs + self.caps_sigma * torch.randn_like(obs)
            loss = loss + self.caps_s * (mean - self.actor.deterministic(noisy)).pow(2).sum(-1).mean()
        return loss

    def _step(self) -> None:
        batch = self.replay.sample()
        obs, act, rew, next_obs, done, discount = batch[:6]
        critic_obs, critic_next = obs, next_obs
        if len(batch) > 6:  # asymmetric critic: clean state and physical parameters, critic only
            critic_obs, critic_next = torch.cat((obs, batch[6]), dim=1), torch.cat((next_obs, batch[7]), dim=1)
        pi, log_prob = self.actor.sample(obs)

        alpha = self.log_alpha.detach().exp()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=False)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            next_a, next_lp = self.actor.sample(next_obs)
            t0, t1 = self.critic_target(critic_next, next_a)
            target = rew + (1.0 - done) * discount * (self._combine(t0, t1) - alpha * next_lp)

        q0, q1 = self.critic(critic_obs, act)
        critic_loss = 0.5 * (F.mse_loss(q0, target) + F.mse_loss(q1, target))
        self.critic_opt.zero_grad(set_to_none=False)
        critic_loss.backward()
        self.critic_opt.step()

        q0_pi, q1_pi = self.critic(critic_obs, pi)
        actor_loss = (alpha * log_prob - self._combine(q0_pi, q1_pi)).mean()
        if self.caps_t or self.caps_s:
            actor_loss = actor_loss + self._smoothness(obs, next_obs, done)
        self.actor_opt.zero_grad(set_to_none=False)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            torch._foreach_mul_(self._t, 1.0 - self.tau)
            torch._foreach_add_(self._t, self._p, alpha=self.tau)
            if self.actor_ema is not None:
                torch._foreach_mul_(self._e, 1.0 - self.ema_tau)
                torch._foreach_add_(self._e, self._a, alpha=self.ema_tau)
            self.stats.copy_(torch.stack((critic_loss.detach(), actor_loss.detach(), alpha_loss.detach(), alpha.squeeze(), log_prob.detach().mean())))

    def update(self) -> None:
        if self.mode != "graph":
            self._step()
        else:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        self.n_updates += 1

    def _capture(self) -> None:
        self._step()  # allocates gradients and optimizer state at fixed addresses
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step()

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mode != "graph":
            return self.actor.sample(obs)[0]
        if self._act_graph is None:
            self.actor.sample(obs)  # BLAS handles must exist before capture
            self._act_in = torch.zeros_like(obs)
            self._act_out = torch.zeros((obs.shape[0], self.replay.act_dim), device=self.device)
            self._act_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._act_graph):
                self._act_out.copy_(self.actor.sample(self._act_in)[0])
        self._act_in.copy_(obs)
        self._act_graph.replay()
        return self._act_out

    def read_stats(self) -> dict:
        s = self.stats.tolist()
        return {"critic_loss": s[0], "actor_loss": s[1], "alpha_loss": s[2], "alpha": s[3], "log_prob": s[4]}

    def parameter_digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for net in (self.actor, self.critic, self.critic_target):
            for p in net.state_dict().values():
                h.update(p.detach().cpu().numpy().tobytes())
        h.update(self.log_alpha.detach().cpu().numpy().tobytes())
        return h.hexdigest()
