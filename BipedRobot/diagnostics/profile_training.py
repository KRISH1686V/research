"""Bounded SB3/HER throughput profiles for the active trainer configuration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from BipedRobot.envs.her_loco_env import HERLocoEnv


class TimingCallback(BaseCallback):
    def __init__(self, timing: dict[str, float], verbose: int = 0):
        super().__init__(verbose)
        self.timing = timing

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.timing.get("learning_starts", float("inf")):
            self.timing.setdefault(
                "post_warmup_elapsed_start",
                time.perf_counter() - self.timing["profile_started"],
            )
        for info in self.locals.get("infos", []):
            elapsed = info.pop("_profile_env_step_seconds", None)
            if elapsed is not None:
                self.timing["worker_env_step_seconds"] += float(elapsed)
                self.timing["env_step_samples"] += 1.0
        started = time.perf_counter()
        self.timing["callbacks_seconds"] += time.perf_counter() - started
        self.timing["callback_calls"] += 1
        return True


def make_env(config: dict[str, Any], seed: int, worker_threads: int | None):
    def _make():
        if worker_threads is not None:
            torch.set_num_threads(worker_threads)
        env = HERLocoEnv(config=copy.deepcopy(config), render_mode=None)
        env.reset(seed=seed)
        original_step = env.step

        def timed_step(action):
            started = time.perf_counter()
            result = original_step(action)
            observation, reward, terminated, truncated, info = result
            info["_profile_env_step_seconds"] = time.perf_counter() - started
            return observation, reward, terminated, truncated, info

        env.step = timed_step
        return env

    return _make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--gradient_steps", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--worker_threads", type=int, default=None)
    parser.add_argument("--copy_info_dict", type=int, choices=[0, 1], default=1)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--main_threads", type=int, default=None)
    parser.add_argument("--cuda_sync", action="store_true")
    parser.add_argument("--output_root", default="paper_results/throughput_profiles")
    args = parser.parse_args()
    if args.main_threads is not None:
        torch.set_num_threads(args.main_threads)

    with open(args.config, encoding="utf-8") as handle:
        profile_config = yaml.safe_load(handle)
    base_path = profile_config.get("base_config")
    if base_path:
        base_path = (Path(args.config).parent / base_path).resolve()
        with base_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for section, values in profile_config.get("overrides", {}).items():
            config.setdefault(section, {}).update(values)
    else:
        config = profile_config
    env_config = copy.deepcopy(config)
    for section in (
        "env",
        "robot",
        "action",
        "obs_scale",
        "actuator",
        "domain_randomization",
        "termination",
        "goal_ranges",
        "curriculum",
        "reward",
        "shaping",
        "budget_goal",
    ):
        if section in config:
            env_config[section] = copy.deepcopy(config[section])
    env_config.update(config.get("env", {}))
    env_config["max_episode_steps"] = int(config.get("env", {}).get("max_episode_steps", 500))
    env_config["realism"] = "easy"
    env_config["push_enabled"] = False

    output = Path(args.output_root) / args.exp_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    timing: dict[str, float] = {
        "vec_step_seconds": 0.0,
        "worker_env_step_seconds": 0.0,
        "env_step_samples": 0.0,
        "her_sample_seconds": 0.0,
        "her_sample_calls": 0.0,
        "her_env_method_seconds": 0.0,
        "her_env_method_calls": 0.0,
        "her_copy_seconds": 0.0,
        "her_copy_calls": 0.0,
        "gradient_seconds": 0.0,
        "gradient_calls": 0.0,
        "callbacks_seconds": 0.0,
        "callback_calls": 0.0,
        "learning_starts": float(args.learning_starts),
        "critic_update_seconds": 0.0,
        "actor_update_seconds": 0.0,
        "alpha_update_seconds": 0.0,
    }

    env = SubprocVecEnv(
        [make_env(env_config, 100 + i, args.worker_threads) for i in range(args.n_envs)],
        start_method="fork",
    )
    original_vec_step = env.step

    def timed_vec_step(actions):
        started = time.perf_counter()
        result = original_vec_step(actions)
        timing["vec_step_seconds"] += time.perf_counter() - started
        return result

    env.step = timed_vec_step
    original_env_method = env.env_method

    def timed_env_method(method_name, *method_args, **method_kwargs):
        started = time.perf_counter()
        result = original_env_method(method_name, *method_args, **method_kwargs)
        timing["her_env_method_seconds"] += time.perf_counter() - started
        timing["her_env_method_calls"] += 1.0
        return result

    env.env_method = timed_env_method

    # These patches are scoped to SB3 HER's copy and reward call paths.
    import stable_baselines3.her.her_replay_buffer as her_module

    original_deepcopy = her_module.copy.deepcopy

    def timed_deepcopy(value, memo=None):
        started = time.perf_counter()
        if memo is None:
            result = original_deepcopy(value)
        else:
            result = original_deepcopy(value, memo)
        timing["her_copy_seconds"] += time.perf_counter() - started
        timing["her_copy_calls"] += 1.0
        return result

    her_module.copy.deepcopy = timed_deepcopy

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": int(config.get("her", {}).get("n_sampled_goal", 2)),
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
            "copy_info_dict": bool(args.copy_info_dict),
        },
        batch_size=int(config.get("sac", {}).get("batch_size", 256)),
        # A complete episode is required before SB3 HER can sample. This
        # profile-only warmup avoids measuring the known early-abort path.
        learning_starts=args.learning_starts,
        train_freq=1,
        gradient_steps=args.gradient_steps,
        device=args.device,
        seed=0,
        verbose=0,
    )
    original_sample = model.replay_buffer.sample
    original_train = model.train

    def timed_sample(*sample_args, **sample_kwargs):
        started = time.perf_counter()
        result = original_sample(*sample_args, **sample_kwargs)
        timing["her_sample_seconds"] += time.perf_counter() - started
        timing["her_sample_calls"] += 1.0
        return result

    def timed_train(*train_args, **train_kwargs):
        started = time.perf_counter()
        result = original_train(*train_args, **train_kwargs)
        if args.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["gradient_seconds"] += time.perf_counter() - started
        timing["gradient_calls"] += 1.0
        return result

    model.replay_buffer.sample = timed_sample
    model.train = timed_train

    def timed_optimizer_step(optimizer, label):
        original_step = optimizer.step

        def step(*step_args, **step_kwargs):
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_step(*step_args, **step_kwargs)
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing[f"{label}_update_seconds"] += time.perf_counter() - started
            return result

        optimizer.step = step

    timed_optimizer_step(model.critic.optimizer, "critic")
    timed_optimizer_step(model.actor.optimizer, "actor")
    timed_optimizer_step(model.ent_coef_optimizer, "alpha")

    if args.cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    timing["profile_started"] = started
    model.learn(args.steps, callback=TimingCallback(timing))
    if args.cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    env.close()
    her_module.copy.deepcopy = original_deepcopy

    timing["elapsed_seconds"] = elapsed
    timing["steps_per_second"] = args.steps / max(elapsed, 1e-9)
    if "post_warmup_started" in timing:
        timing["post_warmup_steps_per_second"] = (
            max(0, args.steps - args.learning_starts)
            / max(elapsed - timing["post_warmup_elapsed_start"], 1e-9)
        )
    elif "post_warmup_elapsed_start" in timing:
        timing["post_warmup_steps_per_second"] = (
            max(0, args.steps - args.learning_starts)
            / max(elapsed - timing["post_warmup_elapsed_start"], 1e-9)
        )
    timing.pop("profile_started", None)
    timing["actual_gradient_updates"] = int(model._n_updates)
    timing["sample_residual_seconds"] = max(
        0.0,
        timing["her_sample_seconds"]
        - timing["her_env_method_seconds"]
        - timing["her_copy_seconds"],
    )
    timing["compute_reward_batch_vectorized"] = False
    timing["train_freq_steps"] = 1
    timing["gradient_steps_setting"] = args.gradient_steps
    timing["effective_update_to_data_ratio"] = (
        (args.gradient_steps if args.gradient_steps >= 0 else args.n_envs)
        / args.n_envs
    )
    timing["batch_size"] = int(config.get("sac", {}).get("batch_size", 256))
    timing["device"] = str(model.device)
    timing["torch_threads_main"] = torch.get_num_threads()
    timing["torch_threads_worker"] = args.worker_threads or "inherited"
    timing["vec_env"] = "SubprocVecEnv(fork)"
    timing["copy_info_dict"] = bool(args.copy_info_dict)
    timing["compute_reward_vectorized"] = False
    timing["mean_info_bytes"] = None
    (output / "profile.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
