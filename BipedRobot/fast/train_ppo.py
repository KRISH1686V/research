"""PPO baseline for E-B: the same env, reward and sensors as the deploy configs, with SB3's PPO. No HER, no mirror, no critic-only inputs.

Use it with budgets switched off (config/v2_deploy_cmd*.yaml). Saves the policy zip and its observation normalizer every --save_every steps and at the end.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import copy
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.collector import env_class, flatten_obs  # noqa: E402
from BipedRobot.fast.nets import actor_inputs  # noqa: E402
from BipedRobot.train_her_biped import _write_run_metadata, build_env_config  # noqa: E402


class SensorEnv(gym.Wrapper):
    """The env as the policy sees it: one flat vector of what the robot can measure."""

    def __init__(self, env, keep):
        super().__init__(env)
        self.keep = np.asarray(keep)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.keep),), np.float32)

    def reset(self, **kwargs):
        o, info = self.env.reset(**kwargs)
        return flatten_obs(o)[self.keep], info

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        return flatten_obs(o)[self.keep], r, term, trunc, info


def make_env(env_config, keep):
    def build():
        env = env_class(env_config)(config=copy.deepcopy(env_config), render_mode=None)
        env.reward_mode = "track"
        return SensorEnv(env, keep)
    return build


class Progress(BaseCallback):
    def __init__(self, args, run_dir, ckpt_dir, keep):
        super().__init__()
        self.a, self.ckpt_dir, self.keep = args, ckpt_dir, keep
        self.csv = open(run_dir / "progress.csv", "w")
        self.csv.write("steps,wall_s,env_steps_per_s,ep_len_mean,ep_rew_mean\n")
        self.t0 = time.perf_counter()
        self.next_save = args.save_every

    def save(self, stem):
        self.model.save(self.ckpt_dir / stem)
        norm = self.model.get_vec_normalize_env()
        np.savez(self.ckpt_dir / f"{stem}_norm.npz", mean=norm.obs_rms.mean, var=norm.obs_rms.var, clip=norm.clip_obs, keep=self.keep)

    def _on_step(self):
        if self.num_timesteps >= self.next_save and self.num_timesteps < self.a.total_steps:
            self.save(f"ppo_{self.num_timesteps}_steps")
            self.next_save += self.a.save_every
        return True

    def _on_rollout_start(self):
        if self.a.difficulty_ramp:
            self.training_env.env_method("set_difficulty", min(1.0, self.num_timesteps / max(1.0, self.a.total_steps * self.a.difficulty_ramp)))
        wall = time.perf_counter() - self.t0
        info = self.model.ep_info_buffer
        if info:
            self.csv.write(f"{self.num_timesteps},{wall:.1f},{self.num_timesteps / max(wall, 1e-6):.1f},{np.mean([e['l'] for e in info]):.1f},{np.mean([e['r'] for e in info]):.2f}\n")
            self.csv.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=3_000_000)
    p.add_argument("--realism", choices=["easy", "medium", "full"], default="full")
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--n_steps", type=int, default=2048, help="steps per env per update")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--difficulty_ramp", type=float, default=0.4)
    p.add_argument("--save_every", type=int, default=250_000)
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    env_config = build_env_config(config, args.realism, False)
    for section in ("validity", "token_budget"):
        if config.get(section):
            env_config[section] = copy.deepcopy(config[section])
    run_dir = Path(args.log_dir) / args.exp_name / f"seed_{args.seed}"
    ckpt_dir = Path(args.checkpoint_dir) / args.exp_name / f"seed_{args.seed}"
    if run_dir.exists() or ckpt_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {args.exp_name}")
    run_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    meta = argparse.Namespace(**vars(args), algo="ppo", reward_mode="track", relabel_strategy="none", n_sampled_goal=0, stand_only=False)
    _write_run_metadata(str(run_dir), config, env_config, meta)

    probe = env_class(env_config)(config=copy.deepcopy(env_config), render_mode=None)
    if env_config.get("require_xml") and probe.xml_path != os.path.abspath(env_config["require_xml"]):
        raise RuntimeError(f"Loaded model {probe.xml_path}, expected {env_config['require_xml']}.")
    obs_dim = sum(int(np.prod(s.shape)) for s in probe.observation_space.spaces.values())
    probe.close()
    keep = actor_inputs(obs_dim, 6, 3, {"achieved", "levels"})
    print(f"torch device: cpu, PPO, n_envs {args.n_envs}, policy inputs {len(keep)} of {obs_dim}", flush=True)

    torch.set_num_threads(2)
    venv = VecNormalize(VecMonitor(SubprocVecEnv([make_env(env_config, keep) for _ in range(args.n_envs)], start_method="fork")), gamma=float(config["sac"]["gamma"]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=float(config["sac"]["gamma"]), gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=1.0, target_kl=0.02, seed=args.seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh, log_std_init=-1.5))
    cb = Progress(args, run_dir, ckpt_dir, keep)
    model.learn(total_timesteps=args.total_steps, callback=cb)
    cb.save("ppo_final")
    venv.close()
    print(f"Training completed: {ckpt_dir / 'ppo_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
