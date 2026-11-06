"""Env workers over shared memory. One pipe byte per step per worker; no infos or observation dicts cross the process boundary.

Must be constructed before CUDA is initialised, because workers are forked. Stepping is asynchronous, so the learner can
run while workers step.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
from multiprocessing.connection import Connection

import numpy as np

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.fast.budget_env import BudgetLocoEnv
from BipedRobot.fast.valid_env import ValidLocoEnv

SCALARS = ("reward", "terminated", "truncated", "r_smooth", "r_gait_move", "r_gait_still", "r_flight", "fell", "r_track", "torque_saturation_ratio", "cost_flight", "cost_slip", "cost_torque", "cost_extra", "cost_extra2", "cost_extra3")


NO_COST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # room for up to six budget costs, in token_budget.costs order


def _pad_cost(cost) -> tuple:
    cost = tuple(float(x) for x in cost)
    if len(cost) > len(NO_COST):
        raise ValueError(f"at most {len(NO_COST)} budget costs, got {len(cost)}")
    return cost + (0.0,) * (len(NO_COST) - len(cost))


def env_class(env_config: dict):
    """BudgetLocoEnv for token budgets, ValidLocoEnv for realism corrections, otherwise the plain env."""
    if env_config.get("token_budget"):
        return BudgetLocoEnv
    return ValidLocoEnv if env_config.get("validity") else HERLocoEnv


def flatten_obs(o: dict) -> np.ndarray:
    return np.concatenate((o["achieved_goal"], o["desired_goal"], o["observation"]))


def _shared(ctx, shape) -> tuple:
    raw = ctx.RawArray("f", int(np.prod(shape)))
    return raw, np.frombuffer(raw, dtype=np.float32).reshape(shape)


def _worker(i: int, env_config: dict, seed: int, reward_mode: str, conn: Connection, raws: dict, shapes: dict, ctrl, priv_dim: int = 0) -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    env = env_class(env_config)(config=copy.deepcopy(env_config), render_mode=None)
    env.reward_mode = reward_mode
    env.use_friction_focus = True  # training only; test tools build their env without this
    view = {k: np.frombuffer(raws[k], dtype=np.float32).reshape(shapes[k]) for k in raws}
    o, _ = env.reset(seed=seed)
    view["obs"][i] = flatten_obs(o)
    if priv_dim:
        view["priv"][i] = env.privileged()
    conn.send_bytes(b"r")
    n_obs = shapes["obs"][1]
    scale = 1.0
    difficulty = 1.0
    while True:
        if conn.recv_bytes() == b"q":
            break
        if ctrl[1] != scale:
            scale = ctrl[1]
            env.set_curriculum_scale(scale)
        env.set_smoothness_ramp(ctrl[0])
        if ctrl[2] != difficulty:
            difficulty = ctrl[2]
            env.set_difficulty(difficulty)
        o, r, term, trunc, info = env.step(view["action"][i])
        nxt = flatten_obs(o)
        row = view["transition"][i]
        row[:n_obs] = nxt
        row[n_obs:] = (r, term, trunc, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"],
                       info["fell"], info["r_track"], info["torque_saturation_ratio"], *_pad_cost(info.get("budget_cost", NO_COST)))
        if priv_dim:
            view["next_priv"][i] = env.privileged()
        if term or trunc:
            o, _ = env.reset()
            nxt = flatten_obs(o)
        view["obs"][i] = nxt
        if priv_dim:
            view["priv"][i] = env.privileged()
        conn.send_bytes(b"d")
    env.close()


class Collector:
    def __init__(self, env_config: dict, n_envs: int, seed: int, obs_dim: int, act_dim: int, reward_mode: str = "track", priv_dim: int = 0):
        ctx = mp.get_context("fork")
        self.n_envs, self.obs_dim = n_envs, obs_dim
        shapes = {"obs": (n_envs, obs_dim), "action": (n_envs, act_dim), "transition": (n_envs, obs_dim + len(SCALARS))}
        if priv_dim:
            shapes["priv"] = shapes["next_priv"] = (n_envs, priv_dim)
        raws, self.view = {}, {}
        for k, s in shapes.items():
            raws[k], self.view[k] = _shared(ctx, s)
        self.ctrl = ctx.RawArray("d", [0.0, 1.0, 1.0])
        self.conns, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(i, env_config, seed + i, reward_mode, child, raws, shapes, self.ctrl, priv_dim), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:
            c.recv_bytes()
        self.obs = self.view["obs"]
        self.transition = self.view["transition"]

    def set_ramp(self, ramp: float) -> None:
        self.ctrl[0] = ramp

    def set_difficulty(self, d: float) -> None:
        self.ctrl[2] = d

    def set_curriculum_scale(self, scale: float) -> None:
        self.ctrl[1] = scale

    def step_async(self, actions: np.ndarray) -> None:
        self.view["action"][:] = actions
        for c in self.conns:
            c.send_bytes(b"s")

    def step_wait(self) -> None:
        for c in self.conns:
            c.recv_bytes()

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send_bytes(b"q")
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
