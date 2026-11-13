"""Train SAC+HER on HERLocoEnv with the GPU replay and learner. Same layout under runs/ and checkpoints/ as train_her_biped.py."""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import copy
import csv
import hashlib
import json
import resource
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import SCALARS, Collector, env_class  # noqa: E402
from BipedRobot.fast.export_sb3 import Sb3Exporter  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.mirror import action_perm, obs_maps, priv_maps  # noqa: E402
from BipedRobot.fast.nets import actor_inputs  # noqa: E402
from BipedRobot.fast.replay import GpuHerReplay, RewardConfig  # noqa: E402
from BipedRobot.train_her_biped import _write_run_metadata, build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HASHED = ["fast/__init__.py", "fast/collector.py", "fast/export_sb3.py", "fast/learner.py", "fast/nets.py", "fast/replay.py", "fast/train.py", "fast/valid_env.py", "fast/budget_env.py", "fast/budget_replay.py", "fast/gait_costs.py", "fast/mirror.py",
          "envs/her_loco_env.py", "envs/rewards/shared_reward.py"]
# 12 Nov 2026: files of a second robot, hashed only for runs whose config names it in env.robot_type.
ROBOT_HASHED = {"op3": ["envs/op3/op3_env.py", "envs/op3/mirror.py", "envs/op3/make_train_xml.py", "envs/assets/robotis_op3/op3_train.xml", "envs/assets/robotis_op3/scene_train.xml"]}


def _mirror_maps(env_config: dict) -> tuple:
    """(action_perm, obs_maps, priv_maps): the biped's from fast/mirror.py, or the OP3's (envs/op3/mirror.py) when env.robot_type is "op3" (12 Nov 2026)."""
    if env_config.get("robot_type") == "op3":
        from BipedRobot.envs.op3 import mirror as op3_mirror
        return op3_mirror.action_perm, op3_mirror.obs_maps, op3_mirror.priv_maps
    return action_perm, obs_maps, priv_maps


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--realism", choices=["easy", "medium", "full"], default="easy")
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["eager", "fused", "graph"], default="graph")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--updates_per_env_step", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n_step", type=int, default=1, help="FastSAC: n-step returns")
    p.add_argument("--layer_norm", action="store_true", help="FastSAC: LayerNorm in the critics")
    p.add_argument("--avg_q", action="store_true", help="FastSAC: mean of the twin critics instead of their min")
    p.add_argument("--init_log_std", type=float, default=None, help="FastSAC: small initial policy noise")
    p.add_argument("--init_alpha", type=float, default=1.0, help="initial entropy coefficient")
    p.add_argument("--caps_t", type=float, default=0.0, help="policy smoothness loss between consecutive states (CAPS); a training loss, not a reward")
    p.add_argument("--caps_s", type=float, default=0.0, help="policy smoothness loss between a state and a noisy copy (CAPS)")
    p.add_argument("--caps_sigma", type=float, default=0.01, help="noise for --caps_s, on the scaled inputs")
    p.add_argument("--asym", action="store_true", help="asymmetric critic: clean state and physical parameters, critic only")
    p.add_argument("--ema_tau", type=float, default=0.0, help="saved policy is an exponential average of the actor's weights")
    p.add_argument("--actor_hide", default="", help="comma list of achieved,levels: inputs only the critic sees, because a real robot cannot measure them")
    p.add_argument("--mirror", action="store_true", help="left-right mirror augmentation (V2 robot)")
    p.add_argument("--difficulty_ramp", type=float, default=0.0, help="fraction of training over which randomization ramps from easy to configured")
    p.add_argument("--plain_ckpt", action="store_true", help="save plain weights instead of SB3 zips")
    p.add_argument("--save_critic", action="store_true", help="plain checkpoints also hold the critic, for reading budget prices from it later")
    p.add_argument("--log_every", type=int, default=20_000)
    p.add_argument("--save_every", type=int, default=100_000)
    p.add_argument("--max_wall_s", type=float, default=None, help="stop early at this wall time; the final checkpoint and summary are still written")
    return p.parse_args()


def main() -> None:
    args = _args()
    config = yaml.safe_load(open(args.config))
    env_config = build_env_config(config, args.realism, False)
    for section in ("validity", "token_budget"):
        if config.get(section):
            env_config[section] = copy.deepcopy(config[section])
    budget = config.get("token_budget")
    run_dir = Path(args.log_dir) / args.exp_name / f"seed_{args.seed}"
    ckpt_dir = Path(args.checkpoint_dir) / args.exp_name / f"seed_{args.seed}"
    if run_dir.exists() or ckpt_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {args.exp_name}")
    run_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    sac = config["sac"]
    k_goal = int(config["her"]["n_sampled_goal"])
    batch = int(args.batch_size or sac["batch_size"])
    learning_starts = int(sac.get("learning_starts", 5000))
    lr, gamma, tau = float(sac["actor_lr"]), float(sac["gamma"]), float(sac["tau"])
    meta = argparse.Namespace(**vars(args), algo="sac", reward_mode="track", relabel_strategy="future", n_sampled_goal=k_goal, stand_only=False)
    _write_run_metadata(str(run_dir), config, env_config, meta)
    hashed = HASHED + ROBOT_HASHED.get(env_config.get("robot_type"), [])
    (run_dir / "code_hashes.json").write_text(json.dumps({f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in hashed}, indent=1))

    probe = env_class(env_config)(config=copy.deepcopy(env_config), render_mode=None)
    if env_config.get("require_xml") and probe.xml_path != os.path.abspath(env_config["require_xml"]):
        raise RuntimeError(f"Loaded model {probe.xml_path}, expected {env_config['require_xml']}. HERLocoEnv falls back to the old robot when a path does not resolve.")
    probe.reward_mode = "track"
    reward_cfg = RewardConfig.from_env(probe)
    obs_dim = sum(int(np.prod(s.shape)) for s in probe.observation_space.spaces.values())
    act_dim = int(probe.action_space.shape[0])
    frames = int(getattr(probe, "history", 1))
    clock = bool(getattr(probe, "clock_hz", 0.0))
    slow_yaw = bool(getattr(probe, "slow_yaw_s", 0.0))
    priv_dim = int(probe.privileged().shape[0]) if args.asym and probe.reset(seed=args.seed) else 0
    priv_signed = int(getattr(probe, "priv_signed", 0))
    probe.close()
    cost_names = tuple(budget.get("costs", ("flight", "slip", "torque"))) if budget else ()
    n_budget = len(cost_names)
    hide = {h for h in args.actor_hide.split(",") if h}
    if hide - {"achieved", "levels"}:
        raise ValueError(f"unknown --actor_hide {hide}")
    if hide and not args.plain_ckpt:
        raise ValueError("--actor_hide needs --plain_ckpt: SB3 zips cannot hold a masked policy")
    actor_keep = actor_inputs(obs_dim, 3 + n_budget, n_budget, hide) if hide else None
    mirror = None
    if args.mirror:
        act_perm_fn, obs_maps_fn, priv_maps_fn = _mirror_maps(env_config)
        op, os_ = obs_maps_fn(frames, n_budget, clock, slow_yaw)
        mirror = {"obs_perm": op, "obs_sign": os_, "act_perm": act_perm_fn()}
        if priv_dim:
            pp, ps = priv_maps_fn(priv_dim - 42 - priv_signed, priv_signed)
            mirror.update(priv_perm=pp, priv_sign=ps)
    actor_mirror = None
    if config.get("equivariant_actor"):  # the designer's exactly left-right symmetric policy, 9 Nov 2026; off unless the config sets it
        if mirror is None:
            raise ValueError("equivariant_actor needs --mirror")
        actor_mirror = {k: mirror[k] for k in ("obs_perm", "obs_sign", "act_perm")}
    collector = Collector(env_config, args.n_envs, args.seed, obs_dim, act_dim, priv_dim=priv_dim)  # forks: before any CUDA use
    exporter = None if (budget or args.plain_ckpt or frames > 1 or args.asym) else Sb3Exporter(env_config, k_goal, batch, gamma, tau, lr)

    def save(path_stem: str) -> None:
        if exporter:
            exporter.save(learner, str(ckpt_dir / path_stem), steps)
        else:  # goals and observations differ from SB3's env, so budget policies are saved as plain weights
            policy = learner.actor_ema if learner.actor_ema is not None else learner.actor
            state = {"actor": policy.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim, "steps": steps}
            if args.save_critic:
                state.update(critic=learner.critic.state_dict(), priv_dim=priv_dim, layer_norm=args.layer_norm, avg_q=args.avg_q)
            torch.save(state, ckpt_dir / f"{path_stem}.pt")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if budget:
        if args.n_step != 1:
            raise ValueError("n-step returns are not supported with token budgets yet")
        dims = list(config["her"].get("relabel_dims", ["vx", "vy", "wz"]))
        if set(dims) - {"vx", "vy", "wz"}:
            raise ValueError(f"her.relabel_dims {dims}: only vx, vy, wz")
        keep = tuple(i for i, n in enumerate(("vx", "vy", "wz")) if n not in dims)
        replay = BudgetHerReplay(1_000_000, args.n_envs, batch, k_goal, obs_dim, act_dim, reward_cfg, device, budget, int(env_config.get("max_episode_steps", 500)), priv_dim=priv_dim, mirror=mirror,
                                 relabel_keep=keep)
    else:
        if config["her"].get("relabel_dims") not in (None, ["vx", "vy", "wz"]):
            raise ValueError("her.relabel_dims needs token budgets (BudgetHerReplay)")
        replay = GpuHerReplay(1_000_000, args.n_envs, batch, k_goal, obs_dim, act_dim, reward_cfg, device, n_step=args.n_step, gamma=gamma, priv_dim=priv_dim, mirror=mirror)
    learner = SacLearner(replay, obs_dim, act_dim, device, mode=args.mode, lr=lr, gamma=gamma, tau=tau, layer_norm=args.layer_norm,
                         avg_q=args.avg_q, init_log_std=args.init_log_std, init_alpha=args.init_alpha, priv_dim=priv_dim, ema_tau=args.ema_tau, actor_keep=actor_keep,
                         caps_t=args.caps_t, caps_s=args.caps_s, caps_sigma=args.caps_sigma, actor_mirror=actor_mirror,
                         target_entropy=config.get("sac_target_entropy"))
    print(f"torch device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}) mode: {args.mode} "
          f"threads: {torch.get_num_threads()} n_envs: {args.n_envs} batch: {batch} updates/env step: {args.updates_per_env_step} "
          f"realism: {args.realism} load: {os.getloadavg()}", flush=True)

    n = args.n_envs
    idx = {name: obs_dim + i for i, name in enumerate(SCALARS)}
    row_np = np.zeros((n, replay.width), dtype=np.float32)
    row_gpu = torch.zeros((n, replay.width), device=device)
    obs_gpu = torch.zeros((n, obs_dim), device=device)
    if args.difficulty_ramp:
        collector.set_difficulty(0.0)
        if budget and budget.get("curriculum"):
            replay.set_budget_floor(1.0)

    curriculum = config.get("curriculum", {})
    shaping = env_config.get("shaping", {})
    ramp_fraction = float(shaping.get("smoothness_ramp_fraction", 0.4))
    successes: deque = deque(maxlen=int(curriculum.get("window_size", 30)))
    scale = 1.0
    ep_track, ep_len, ep_rew = np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n)
    ep_fell = np.zeros(n, dtype=bool)
    recent_len, recent_rew = deque(maxlen=100), deque(maxlen=100)
    T = {k: 0.0 for k in ("act", "step_async", "updates", "step_wait", "replay_add", "bookkeeping", "export_log")}

    csv_file = open(run_dir / "progress.csv", "w", newline="")
    writer = None
    steps, debt, next_log, next_save = 0, 0.0, args.log_every, args.save_every
    t_start = t_mark = time.perf_counter()
    t_post, cpu_post, steps_post, upd_post, last_steps, last_upd = None, 0.0, 0, 0, 0, 0
    pc = time.perf_counter
    lagrange = budget.get("lagrange") if budget else None
    dual_every = int(lagrange.get("every", 1000)) if lagrange else 0
    dual_cost, dual_n = np.zeros(n_budget), 0
    lam_now = replay._lam_np.copy() if lagrange else None
    cost_sum, cost_n = np.zeros(n_budget), 0

    while steps < args.total_steps:
        t0 = pc()
        if args.max_wall_s is not None and t0 - t_start > args.max_wall_s:
            print(f"Stopping at max_wall_s after {steps} env steps", flush=True)
            break
        prev_obs = collector.obs.copy()
        prev_priv = collector.view["priv"].copy() if priv_dim else None
        if steps < learning_starts:
            actions = rng.uniform(-1.0, 1.0, (n, act_dim)).astype(np.float32)
        else:
            obs_gpu.copy_(torch.from_numpy(prev_obs))
            actions = learner.act(obs_gpu).cpu().numpy()
        t1 = pc()
        collector.step_async(actions)
        t2 = pc()
        todo = int(debt)
        debt -= todo
        for _ in range(todo):
            learner.update()
        t3 = pc()
        collector.step_wait()
        t4 = pc()
        tr = collector.transition
        term, trunc = tr[:, idx["terminated"]] > 0.5, tr[:, idx["truncated"]] > 0.5
        ended = term | trunc
        row_np[:, :obs_dim] = prev_obs
        row_np[:, replay.i_next:replay.i_act] = tr[:, :obs_dim]
        row_np[:, replay.i_act:replay.i_rew] = actions
        i = replay.i_rew
        row_np[:, i] = tr[:, idx["reward"]]
        row_np[:, i + 1] = term
        row_np[:, i + 2] = tr[:, idx["r_smooth"]]
        row_np[:, i + 3] = tr[:, idx["r_gait_move"]]
        row_np[:, i + 4] = tr[:, idx["r_gait_still"]]
        row_np[:, i + 5] = tr[:, idx["r_flight"]]
        row_np[:, i + 6] = tr[:, idx["fell"]]
        if budget:
            row_np[:, i + 7:i + 7 + n_budget] = tr[:, idx["cost_flight"]:idx["cost_flight"] + n_budget]
        if priv_dim:
            row_np[:, replay.i_priv:replay.i_next_priv] = prev_priv
            row_np[:, replay.i_next_priv:replay.i_next_priv + priv_dim] = collector.view["next_priv"]
        row_gpu.copy_(torch.from_numpy(row_np))
        if budget:
            replay.add(row_gpu, ended, row_np[:, i + 7:i + 7 + n_budget])
        else:
            replay.add(row_gpu, ended)
        steps += n
        if steps > learning_starts:
            if t_post is None:
                t_post, steps_post, upd_post = pc(), steps, learner.n_updates
                r = resource.getrusage(resource.RUSAGE_SELF)
                cpu_post = r.ru_utime + r.ru_stime
                T = {k: 0.0 for k in T}
            debt += n * args.updates_per_env_step
        t5 = pc()

        # Same curriculum and ramp rules as StageACallback.
        if budget:
            cost_sum += tr[:, idx["cost_flight"]:idx["cost_flight"] + n_budget].sum(0)
            cost_n += n
        if lagrange:
            dual_cost += tr[:, idx["cost_flight"]:idx["cost_flight"] + n_budget].sum(0)
            dual_n += n
            if dual_n >= dual_every and steps > learning_starts:
                lam_now = replay.dual_step(dual_cost / dual_n)
                dual_cost, dual_n = np.zeros(n_budget), 0
        ep_track += tr[:, idx["r_track"]]
        ep_rew += tr[:, idx["reward"]]
        ep_len += 1
        ep_fell |= tr[:, idx["fell"]] > 0.5
        if ended.any():
            for e in np.flatnonzero(ended):
                if np.linalg.norm(tr[e, 3:6]) > 1e-8:
                    successes.append(float(not ep_fell[e] and ep_track[e] / max(1, ep_len[e]) >= 0.6))
                recent_len.append(int(ep_len[e]))
                recent_rew.append(float(ep_rew[e]))
                ep_track[e], ep_rew[e], ep_len[e], ep_fell[e] = 0.0, 0.0, 0, False
            if curriculum.get("enabled", False) and len(successes) == successes.maxlen \
                    and np.mean(successes) >= float(curriculum.get("success_threshold", 0.7)):
                scale *= float(curriculum.get("expansion_rate", 1.15))
                collector.set_curriculum_scale(scale)
                successes.clear()
        collector.set_ramp(min(1.0, steps / max(1.0, args.total_steps * ramp_fraction)))
        if args.difficulty_ramp:
            d = min(1.0, steps / max(1.0, args.total_steps * args.difficulty_ramp))
            collector.set_difficulty(d)
            if budget and budget.get("curriculum"):
                replay.set_budget_floor(1.0 - d * (1.0 - float(budget.get("u_low", 0.0))))  # u_low < 0: training also draws budgets tighter than u = 0
        t6 = pc()

        if steps >= next_save and steps < args.total_steps:
            save(f"sb3_her_model_{steps}_steps")
            next_save += args.save_every
        if steps >= next_log:
            now = pc()
            row = {"steps": steps, "updates": learner.n_updates, "wall_s": round(now - t_start, 1),
                   "env_steps_per_s": round((steps - last_steps) / (now - t_mark), 1),
                   "updates_per_s": round((learner.n_updates - last_upd) / (now - t_mark), 1),
                   "ep_len_mean": round(float(np.mean(recent_len)), 1) if recent_len else None,
                   "ep_rew_mean": round(float(np.mean(recent_rew)), 2) if recent_rew else None,
                   "range_scale": scale, "load1": round(os.getloadavg()[0], 1),
                   **{k: round(v, 4) for k, v in learner.read_stats().items()}}
            if budget:
                row.update(zip(tuple("cost_" + c for c in cost_names), (round(float(x), 5) for x in cost_sum / max(cost_n, 1))))
                cost_sum, cost_n = np.zeros(n_budget), 0
            if lagrange:
                row.update(zip(tuple("lambda_" + c for c in cost_names), (round(float(x), 4) for x in lam_now)))
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            print(json.dumps(row), flush=True)
            t_mark, last_steps, last_upd = now, steps, learner.n_updates
            next_log += args.log_every
        t7 = pc()
        for key, dt in (("act", t1 - t0), ("step_async", t2 - t1), ("updates", t3 - t2), ("step_wait", t4 - t3),
                        ("replay_add", t5 - t4), ("bookkeeping", t6 - t5), ("export_log", t7 - t6)):
            T[key] += dt

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = pc()
    collector.close()
    save("sb3_her_final")
    r = resource.getrusage(resource.RUSAGE_SELF)
    wall_post = end - t_post if t_post else None
    summary = {
        "device": str(device), "mode": args.mode, "n_envs": n, "batch": batch, "updates_per_env_step": args.updates_per_env_step,
        "steps": steps, "updates": learner.n_updates, "wall_s": end - t_start, "wall_post_warmup_s": wall_post,
        "env_steps_per_s_post_warmup": (steps - steps_post) / wall_post if wall_post else None,
        "updates_per_s_post_warmup": (learner.n_updates - upd_post) / wall_post if wall_post else None,
        "main_cores_used_post_warmup": (r.ru_utime + r.ru_stime - cpu_post) / wall_post if wall_post else None,
        "sections_s_post_warmup": {k: round(v, 3) for k, v in T.items()},
        "gpu_mem_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
        "load_at_end": os.getloadavg(), "parameter_digest": learner.parameter_digest(), "range_scale": scale,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"Training completed: {ckpt_dir / 'sb3_her_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
