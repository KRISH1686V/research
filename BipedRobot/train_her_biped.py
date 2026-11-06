"""Stage A SB3 SAC + HER training entry point."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.her.budget_her_replay_buffer import BudgetHerReplayBuffer
from BipedRobot.envs.rewards.adaptive_reward import AdaptiveRewardFunction
from BipedRobot.evaluation_suite import evaluate_command_suite


def _run_command(command: list[str], *, cwd: str) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def _run_command_bytes(command: list[str], *, cwd: str) -> bytes:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True)
    return completed.stdout


def _write_run_metadata(
    run_dir: str,
    config: Dict[str, Any],
    env_config: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Write all provenance required to reproduce a training run."""
    root = os.path.dirname(os.path.abspath(__file__))
    path = Path(run_dir)
    resolved = copy.deepcopy(config)
    resolved["resolved_env_config"] = copy.deepcopy(env_config)
    resolved["cli"] = vars(args)
    resolved["training"] = {
        "algo": args.algo,
        "reward_mode": args.reward_mode,
        "relabel_strategy": args.relabel_strategy,
        "n_sampled_goal": args.n_sampled_goal,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "realism": args.realism,
        "stand_only": args.stand_only,
        "n_envs": args.n_envs,
    }
    resolved_text = yaml.safe_dump(resolved, sort_keys=True)
    config_hash = hashlib.sha256(resolved_text.encode("utf-8")).hexdigest()

    try:
        commit = _run_command(["git", "rev-parse", "HEAD"], cwd=root).strip()
        diff = _run_command_bytes(["git", "diff", "--binary"], cwd=root)
        tracked = set(_run_command(["git", "ls-files"], cwd=root).splitlines())
        untracked = _run_command(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=root
        ).splitlines()
        imported_untracked = []
        for module in list(sys.modules.values()):
            module_file = getattr(module, "__file__", None)
            if not module_file:
                continue
            try:
                relative = os.path.relpath(module_file, root)
            except ValueError:
                continue
            if relative.startswith("..") or relative in tracked or relative not in untracked:
                continue
            imported_untracked.append(relative)
        files = {
            "resolved_config.yaml": resolved_text,
            "git_commit.txt": commit + "\n",
            "git_diff.patch": diff,
            "untracked_imported_files.txt": "\n".join(sorted(set(imported_untracked))) + "\n",
            "command_line.txt": " ".join(sys.argv) + "\n",
            "seed.txt": f"{args.seed}\n",
            "pip_freeze.txt": _run_command([sys.executable, "-m", "pip", "freeze"], cwd=root),
            "evaluation_config_hash.txt": config_hash + "\n",
        }
        for name, contents in files.items():
            destination = path / name
            if isinstance(contents, bytes):
                destination.write_bytes(contents)
            else:
                destination.write_text(contents, encoding="utf-8")
            if not destination.exists():
                raise OSError(f"metadata file was not created: {destination}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to write complete run provenance in {run_dir}") from exc
    return config_hash


def build_env_config(config: Dict[str, Any], realism: str, stand_only: bool) -> Dict[str, Any]:
    env_config = copy.deepcopy(config.get("env", {}))
    for section in ("robot", "action", "obs_scale", "actuator", "domain_randomization", "termination", "goal_ranges", "curriculum", "reward", "shaping", "budget_goal"):
        if section in config:
            env_config[section] = copy.deepcopy(config[section])
    env_config["realism"] = realism
    env_config["push_enabled"] = realism == "full"
    if stand_only:
        env_config.setdefault("curriculum", {})["enabled"] = False
        env_config["curriculum"]["p_zero_cmd"] = 1.0
    return env_config


def validate_pilot_config(config: Dict[str, Any]) -> None:
    shaping = config.get("shaping", {})
    her = config.get("her", {})
    curriculum = config.get("curriculum", {})
    goal_ranges = config.get("goal_ranges", {})
    required_flags = ("gait_shaping", "action_filter", "prev_action_obs", "smooth_penalties", "reset_velocity")
    missing = [flag for flag in required_flags if flag not in shaping]
    if missing:
        raise ValueError(f"Pilot config missing shaping flags: {', '.join(missing)}")
    if int(her.get("n_sampled_goal", -1)) != 2:
        raise ValueError("Pilot config requires her.n_sampled_goal=2")
    if float(curriculum.get("p_zero_cmd", -1.0)) != 0.20:
        raise ValueError("Pilot config requires curriculum.p_zero_cmd=0.20")
    if float(goal_ranges.get("vx", [0.0])[0]) < 0.20:
        raise ValueError("Pilot config requires minimum nonzero vx >= 0.20")


def make_env_fn(env_config: Dict[str, Any], seed: int, reward_mode: str):
    def _make() -> Monitor:
        env = HERLocoEnv(config=copy.deepcopy(env_config), render_mode=None)
        env.reset(seed=seed)
        env.reward_mode = reward_mode
        if reward_mode == "adaptive":
            env.adaptive_reward_fn = AdaptiveRewardFunction(
                reward_mode="adaptive", config=env_config.get("reward", {})
            )
        return Monitor(env)
    return _make


class StageACallback(BaseCallback):
    def __init__(
        self,
        curriculum_config: Dict[str, Any],
        stand_only: bool,
        eval_env: HERLocoEnv,
        n_envs: int = 1,
        evaluation_config: Dict[str, Any] | None = None,
        artifact_dir: str | None = None,
        total_steps: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.stand_only = stand_only
        self.eval_env = eval_env
        self.evaluation_config = evaluation_config or {}
        self.evaluation_config_hash = hashlib.sha256(
            json.dumps(self.evaluation_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.evaluation_index = 0
        self.monitor_step = 300_000
        self.monitor_failed = False
        self.successes = deque(maxlen=int(curriculum_config.get("window_size", 30)))
        self.range_scale = 1.0
        self.next_eval = 50_000
        self.episode_tracking = np.zeros(n_envs, dtype=np.float64)
        self.episode_steps = np.zeros(n_envs, dtype=np.int64)
        self.episode_fell = np.zeros(n_envs, dtype=bool)
        shaping = eval_env.config.get("shaping", {})
        self.smoothness_ramp_enabled = bool(shaping.get("smooth_penalties", False))
        self.smoothness_ramp_fraction = float(
            shaping.get("smoothness_ramp_fraction", 0.4)
        )
        self.total_steps = max(1, int(total_steps))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = np.asarray(self.locals.get("actions", []))
        if actions.size:
            self.logger.record("train/action_abs_gt_095", float(np.mean(np.abs(actions) > 0.95)))
        if infos:
            sat = [float(info.get("torque_saturation_ratio", 0.0)) for info in infos]
            track = [float(info.get("r_track", 0.0)) for info in infos]
            self.logger.record("train/torque_sat", float(np.mean(sat)))
            self.logger.record("train/tracking_reward", float(np.mean(track)))
            for idx, info in enumerate(infos):
                if idx >= len(self.episode_tracking):
                    continue
                self.episode_tracking[idx] += float(info.get("r_track", 0.0))
                self.episode_steps[idx] += 1
                self.episode_fell[idx] |= bool(info.get("fell", False))
                if info.get("terminal_observation") is not None or info.get("episode") is not None:
                    goal = np.asarray(info.get("desired_goal", np.zeros(3)))
                    if np.linalg.norm(goal) > 1e-8:
                        mean_track = self.episode_tracking[idx] / max(1, self.episode_steps[idx])
                        self.successes.append(float(not self.episode_fell[idx] and mean_track >= 0.6))
                    self.episode_tracking[idx] = 0.0
                    self.episode_steps[idx] = 0
                    self.episode_fell[idx] = False

        if not self.stand_only and self.curriculum_config.get("enabled", False):
            threshold = float(self.curriculum_config.get("success_threshold", 0.7))
            if len(self.successes) == self.successes.maxlen and np.mean(self.successes) >= threshold:
                self.range_scale *= float(self.curriculum_config.get("expansion_rate", 1.15))
                self.training_env.env_method("set_curriculum_scale", self.range_scale)
                self.successes.clear()
            self.logger.record("curriculum/range_scale", self.range_scale)
            self.logger.record("curriculum/success_rate", float(np.mean(self.successes)) if self.successes else 0.0)
        if self.smoothness_ramp_enabled:
            ramp = min(
                1.0,
                float(self.num_timesteps)
                / max(1.0, self.total_steps * self.smoothness_ramp_fraction),
            )
            self.training_env.env_method("set_smoothness_ramp", ramp)
            self.logger.record("shaping/smoothness_ramp", ramp)

        if self.num_timesteps >= self.next_eval:
            evaluation = self._evaluate()
            if (
                self.num_timesteps >= self.monitor_step
                and not self.monitor_failed
                and evaluation["zero"]["mean_episode_length"] < 400.0
            ):
                self.monitor_failed = True
                if self.artifact_dir:
                    (self.artifact_dir / "MONITOR_FAILURE_300K.txt").write_text(
                        "Diagnostic only: eval/ep_len_zero < 400 at the 300k monitor gate; training continued.\n",
                        encoding="utf-8",
                    )
            self.next_eval += 50_000
        return True

    def _evaluate(self) -> Dict[str, Any]:
        metrics = {}
        policy_fn = lambda obs_dict: self.model.predict(obs_dict, deterministic=True)[0]
        for label, command in (("zero", [0.0, 0.0, 0.0]), ("02", [0.2, 0.0, 0.0])):
            suite = evaluate_command_suite(
                policy_fn,
                self.eval_env,
                command,
                range(1000, 1005),
                max_steps=self.eval_env.max_episode_steps,
            )
            metrics[label] = suite
            self.logger.record(f"eval/ep_len_{label}", suite["mean_episode_length"])
            self.logger.record(f"eval/mean_vx_{label}", suite["mean_vx"])
            self.logger.record(f"eval/double_support_{label}", suite["double_support_fraction"])
            self.logger.record(f"eval/peak_swing_height_{label}", suite["mean_peak_swing_foot_height"])
            self.logger.record(f"eval/touchdown_r_{label}", suite["touchdowns_per_second_per_foot"][0])
            self.logger.record(f"eval/touchdown_l_{label}", suite["touchdowns_per_second_per_foot"][1])
        self.logger.record(
            "eval/torque_sat",
            float(np.mean([metrics["zero"]["torque_saturation_ratio"], metrics["02"]["torque_saturation_ratio"]])),
        )
        self.logger.record("eval/rmse_vx", metrics["02"]["rmse_vx"])
        self.logger.record("eval/config_hash_numeric", int(self.evaluation_config_hash[:12], 16) % 1_000_000_000)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "evaluation_index": self.evaluation_index,
                "num_timesteps": int(self.num_timesteps),
                "config_hash": self.evaluation_config_hash,
                "metrics": metrics,
            }
            with (self.artifact_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        self.evaluation_index += 1
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BipedRobot with SB3 SAC + HER")
    parser.add_argument("--algo", choices=["sac", "td3"], default="sac")
    parser.add_argument("--reward_mode", choices=["adaptive", "fixed", "track", "stand_safe", "budget_goal"], default="track")
    parser.add_argument("--relabel_strategy", choices=["future", "final", "episode"], default="future")
    parser.add_argument("--n_sampled_goal", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--config", default=os.path.join(current_dir, "config", "her_loco_config.yaml"))
    parser.add_argument("--log_dir", default=os.path.join(current_dir, "runs"))
    parser.add_argument("--checkpoint_dir", default=os.path.join(current_dir, "checkpoints"))
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--realism", choices=["easy", "medium", "full"], default="easy")
    parser.add_argument("--stand_only", action="store_true")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a short episode horizon so HER can sample during a smoke test.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    if args.exp_name.startswith("pilot_gait_"):
        validate_pilot_config(config)
    env_config = build_env_config(config, args.realism, args.stand_only)
    if args.smoke:
        env_config["max_episode_steps"] = 1
        config.setdefault("sac", {})["learning_starts"] = max(
            1, int(args.n_envs)
        )
    run_dir = os.path.join(args.log_dir, args.exp_name, f"seed_{args.seed}")
    ckpt_dir = os.path.join(args.checkpoint_dir, args.exp_name, f"seed_{args.seed}")
    if os.path.exists(run_dir) or os.path.exists(ckpt_dir):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {args.exp_name}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(ckpt_dir, exist_ok=False)
    evaluation_config_hash = _write_run_metadata(run_dir, config, env_config, args)

    goal_strategy = {
        "future": GoalSelectionStrategy.FUTURE,
        "final": GoalSelectionStrategy.FINAL,
        "episode": GoalSelectionStrategy.EPISODE,
    }[args.relabel_strategy]
    env_fns = [make_env_fn(env_config, args.seed + i, args.reward_mode) for i in range(args.n_envs)]
    if args.n_envs > 1:
        try:
            training_env = SubprocVecEnv(env_fns, start_method="fork")
        except Exception as exc:
            print(f"SubprocVecEnv failed ({exc}); falling back to n_envs=1")
            training_env = DummyVecEnv([env_fns[0]])
    else:
        training_env = DummyVecEnv(env_fns)
    eval_env = HERLocoEnv(config=copy.deepcopy(env_config), render_mode=None)

    sac_cfg = config.get("sac", {})
    replay_buffer_class = (
        BudgetHerReplayBuffer
        if bool(config.get("budget_goal", {}).get("enabled", False))
        else HerReplayBuffer
    )
    n_sampled_goal = int(config.get("her", {}).get("n_sampled_goal", 4))
    if args.n_sampled_goal is not None:
        n_sampled_goal = args.n_sampled_goal
    replay_kwargs = {
        "n_sampled_goal": n_sampled_goal,
        "goal_selection_strategy": goal_strategy,
        "copy_info_dict": True,
    }
    if replay_buffer_class is BudgetHerReplayBuffer:
        budget_cfg = config["budget_goal"]
        replay_kwargs.update(
            p_budget_relabel=float(budget_cfg.get("p_budget_relabel", 0.6)),
            min_budget=budget_cfg.get("min_budget"),
        )
    common = dict(
        policy="MultiInputPolicy",
        env=training_env,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_kwargs,
        learning_rate=float(sac_cfg.get("actor_lr", 3e-4)),
        buffer_size=1_000_000,
        batch_size=int(sac_cfg.get("batch_size", 256)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        learning_starts=int(sac_cfg.get("learning_starts", 5_000)),
        tensorboard_log=run_dir,
        seed=args.seed,
        verbose=1,
    )
    use_sde = bool(sac_cfg.get("use_sde", False))
    sde_sample_freq = int(sac_cfg.get("sde_sample_freq", -1))
    if args.algo == "sac":
        model = SAC(
            ent_coef="auto",
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            train_freq=1,
            gradient_steps=-1 if args.n_envs > 1 else 1,
            **common,
        )
    else:
        model = TD3(train_freq=1, gradient_steps=-1 if args.n_envs > 1 else 1, **common)

    profile = {"sample_seconds": 0.0, "gradient_seconds": 0.0, "sample_calls": 0, "gradient_calls": 0}
    restore_profile_wrappers = None
    if args.profile:
        original_sample = model.replay_buffer.sample
        original_train = model.train

        def timed_sample(*sample_args, **sample_kwargs):
            started = time.perf_counter()
            result = original_sample(*sample_args, **sample_kwargs)
            profile["sample_seconds"] += time.perf_counter() - started
            profile["sample_calls"] += 1
            return result

        def timed_train(*train_args, **train_kwargs):
            started = time.perf_counter()
            result = original_train(*train_args, **train_kwargs)
            profile["gradient_seconds"] += time.perf_counter() - started
            profile["gradient_calls"] += 1
            return result

        model.replay_buffer.sample = timed_sample
        model.train = timed_train

        def restore_profile_wrappers() -> None:
            del model.replay_buffer.sample
            del model.train

    callbacks = [
        StageACallback(
            config.get("curriculum", {}),
            args.stand_only,
            eval_env,
            n_envs=training_env.num_envs,
            evaluation_config={"resolved_env_config": env_config, "realism": args.realism},
            artifact_dir=run_dir,
            total_steps=args.total_steps,
        ),
        CheckpointCallback(save_freq=max(1, 100_000 // training_env.num_envs), save_path=ckpt_dir, name_prefix="sb3_her_model"),
    ]
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Starting {args.total_steps} steps with realism={args.realism}, n_envs={training_env.num_envs}")
    started = time.perf_counter()
    model.learn(total_timesteps=args.total_steps, callback=CallbackList(callbacks))
    elapsed = time.perf_counter() - started
    if args.profile:
        profile["elapsed_seconds"] = elapsed
        profile["steps_per_second"] = float(args.total_steps / max(elapsed, 1e-9))
        profile["env_and_overhead_seconds"] = max(
            0.0,
            elapsed - profile["sample_seconds"] - profile["gradient_seconds"],
        )
        (Path(run_dir) / "profile.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8"
        )
    if restore_profile_wrappers is not None:
        restore_profile_wrappers()
    training_env.close()
    eval_env.close()
    final_model_path = os.path.join(ckpt_dir, "sb3_her_final.zip")
    model.save(final_model_path)
    print(f"Training completed: {final_model_path}")


if __name__ == "__main__":
    main()
