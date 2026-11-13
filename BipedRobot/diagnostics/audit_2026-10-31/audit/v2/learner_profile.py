"""Where does an SB3 SAC+HER update spend its time? Per-component timing with the trainer's real settings.

Same model construction as train_her_biped.py (SubprocVecEnv fork, HerReplayBuffer, copy_info_dict, batch 256,
gradient_steps -1). `ProfiledSAC.train` is SAC.train with timers around each section; the replay buffer and vec env
are wrapped, not edited. Nothing is written under runs/ or checkpoints/.

GPU work is asynchronous, so --sync calls torch.cuda.synchronize around GPU sections to attribute it correctly. That
adds overhead, so the true updates/s comes from a run without --sync.

Usage: learner_profile.py OUT_JSON [--updates 1000] [--config PATH] [--realism easy] [--n-envs 8] [--threads 4] [--device cuda] [--sync]
"""
import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch as th  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.utils import polyak_update  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402
from stable_baselines3.her import her_replay_buffer as her_module  # noqa: E402
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy  # noqa: E402
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer  # noqa: E402

from BipedRobot.train_her_biped import build_env_config, make_env_fn  # noqa: E402

T = defaultdict(float)
ACTIVE = {"on": False, "sync": False}


@contextmanager
def section(name, gpu=False):
    if not ACTIVE["on"]:
        yield
        return
    if gpu and ACTIVE["sync"]:
        th.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if gpu and ACTIVE["sync"]:
        th.cuda.synchronize()
    T[name] += time.perf_counter() - t0


def wrap(obj, attr, name, gpu=False):
    fn = getattr(obj, attr)

    def wrapped(*a, **k):
        with section(name, gpu=gpu):
            return fn(*a, **k)

    setattr(obj, attr, wrapped)


class ProfiledSAC(SAC):
    def train(self, gradient_steps, batch_size=64):
        with section("train_setup"):
            self.policy.set_training_mode(True)
            optimizers = [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
            self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        for gradient_step in range(gradient_steps):
            with section("replay_sample_total"):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with section("actor_forward", gpu=True):
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
            with section("alpha_update", gpu=True):
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                ent_coefs.append(ent_coef.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            with section("critic_target_forward", gpu=True):
                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            with section("critic_forward", gpu=True):
                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * sum(F.mse_loss(q, target_q_values) for q in current_q_values)
                critic_losses.append(critic_loss.item())
            with section("critic_backward_step", gpu=True):
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
            with section("actor_loss_forward", gpu=True):
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())
            with section("actor_backward_step", gpu=True):
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            with section("target_update", gpu=True):
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            if ACTIVE["on"]:
                T["n_updates"] += 1
        with section("logging"):
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.mem, self.stop = [], [], False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], timeout=10)
                u, m = out.decode().strip().split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--config", default="/home/rpuser1/yasho/rl/BipedRobot/config/pilot_gait_min_v2.yaml")
    p.add_argument("--realism", default="easy")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sync", action="store_true")
    a = p.parse_args()

    th.set_num_threads(a.threads)
    ACTIVE["sync"] = a.sync
    config = yaml.safe_load(open(a.config))
    env_config = build_env_config(config, a.realism, False)
    env_fns = [make_env_fn(env_config, 900 + i, "track") for i in range(a.n_envs)]
    venv = SubprocVecEnv(env_fns, start_method="fork")
    sac_cfg = config["sac"]
    learning_starts = int(sac_cfg.get("learning_starts", 5000))
    model = ProfiledSAC(
        policy="MultiInputPolicy", env=venv, replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": int(config["her"]["n_sampled_goal"]),
                              "goal_selection_strategy": GoalSelectionStrategy.FUTURE, "copy_info_dict": True},
        learning_rate=float(sac_cfg["actor_lr"]), buffer_size=1_000_000, batch_size=int(sac_cfg["batch_size"]),
        gamma=float(sac_cfg["gamma"]), tau=float(sac_cfg["tau"]), learning_starts=learning_starts, seed=900, verbose=0,
        ent_coef="auto", train_freq=1, gradient_steps=-1, device=a.device)
    device = str(model.device)
    print("torch device:", device, "| cuda:", th.cuda.get_device_name(0) if th.cuda.is_available() else None, "| threads:", th.get_num_threads(), flush=True)

    rb = model.replay_buffer
    wrap(rb, "_get_real_samples", "  real_samples")
    wrap(rb, "_get_virtual_samples", "  virtual_samples")
    wrap(rb, "_sample_goals", "    sample_goals")
    wrap(rb, "to_torch", "    host_to_gpu_transfer")
    wrap(venv, "env_method", "    compute_reward_env_method")
    real_deepcopy = copy.deepcopy

    class CopyProxy:
        @staticmethod
        def deepcopy(x, memo=None):
            with section("    info_deepcopy"):
                return real_deepcopy(x, memo)

    her_module.copy = CopyProxy
    wrap(venv, "step_async", "env_step_async")
    wrap(venv, "step_wait", "env_step_wait")
    wrap(model, "_sample_action", "policy_inference")
    wrap(rb, "add", "replay_add")
    wrap(model, "train", "train_total")

    gpu = GpuSampler()
    state = {"t0": None, "cpu0": None, "last": 0.0}
    from stable_baselines3.common.callbacks import BaseCallback

    class StartTimers(BaseCallback):
        def _on_step(self):
            if not ACTIVE["on"] and self.num_timesteps >= learning_starts:
                ACTIVE["on"] = True
                state["t0"] = time.perf_counter()
                r = resource.getrusage(resource.RUSAGE_SELF)
                state["cpu0"] = r.ru_utime + r.ru_stime
                state["steps0"] = self.num_timesteps
                gpu.start()
            return True

    print(f"warmup {learning_starts} env steps, then {a.updates} updates; load before: {os.getloadavg()}", flush=True)
    t_all = time.perf_counter()
    model.learn(learning_starts + a.updates, callback=StartTimers(), log_interval=None, progress_bar=False)
    ACTIVE["on"] = False
    if th.cuda.is_available():
        th.cuda.synchronize()
    wall = time.perf_counter() - state["t0"]
    r = resource.getrusage(resource.RUSAGE_SELF)
    cpu = r.ru_utime + r.ru_stime - state["cpu0"]
    gpu.stop = True
    n = T.pop("n_updates")
    per_1000 = {k: v / n * 1000 for k, v in T.items()}
    grad = ["actor_forward", "alpha_update", "critic_target_forward", "critic_forward", "critic_backward_step", "actor_loss_forward", "actor_backward_step", "target_update"]
    res = {
        "device": device, "cuda_name": th.cuda.get_device_name(0) if th.cuda.is_available() else None, "sync": a.sync,
        "threads": th.get_num_threads(), "n_envs": a.n_envs, "batch": int(sac_cfg["batch_size"]), "updates_measured": int(n),
        "wall_s_measured": wall, "updates_per_s": n / wall, "env_steps_per_s": (model.num_timesteps - state["steps0"]) / wall,
        "main_process_cpu_cores_used": cpu / wall, "load_after": os.getloadavg(),
        "gpu_util_mean_pct": float(np.mean(gpu.util)) if gpu.util else None, "gpu_mem_mb_mean": float(np.mean(gpu.mem)) if gpu.mem else None,
        "gpu_util_samples": len(gpu.util), "seconds_per_1000_updates": per_1000,
        "gradient_sections_total": sum(per_1000.get(k, 0) for k in grad), "wall_per_1000_updates": wall / n * 1000,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "seconds_per_1000_updates"}, indent=1))
    print("seconds per 1000 updates:")
    for k, v in sorted(per_1000.items(), key=lambda x: -x[1]):
        print(f"  {k:34s} {v:8.2f}")
    venv.close()


if __name__ == "__main__":
    main()
