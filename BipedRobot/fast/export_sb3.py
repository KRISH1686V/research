"""Save the fast learner as an SB3 SAC zip, so evaluation_suite_v2 and the audit tools load it unchanged."""
from __future__ import annotations

import copy

import torch
from stable_baselines3 import SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from BipedRobot.envs.her_loco_env import HERLocoEnv


class Sb3Exporter:
    def __init__(self, env_config: dict, n_sampled_goal: int, batch_size: int, gamma: float, tau: float, lr: float):
        env = HERLocoEnv(config=copy.deepcopy(env_config), render_mode=None)
        self.model = SAC(
            "MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"n_sampled_goal": n_sampled_goal, "goal_selection_strategy": GoalSelectionStrategy.FUTURE, "copy_info_dict": True},
            learning_rate=lr, buffer_size=1000, batch_size=batch_size, gamma=gamma, tau=tau, learning_starts=0,
            ent_coef="auto", train_freq=1, gradient_steps=1, device="cpu", verbose=0)
        env.close()

    def save(self, learner, path: str, num_timesteps: int) -> None:
        with torch.no_grad():
            pairs = [(learner.actor, self.model.actor)]
            if not learner.layer_norm:  # SB3's critic has no LayerNorm; evaluation only needs the actor
                pairs += [(learner.critic, self.model.critic), (learner.critic_target, self.model.critic_target)]
            for src, dst in pairs:
                dst.load_state_dict({k: v.detach().cpu() for k, v in src.state_dict().items()}, strict=True)
            self.model.log_ent_coef.copy_(learner.log_alpha.detach().cpu())
        self.model.num_timesteps = num_timesteps
        self.model.save(path)
