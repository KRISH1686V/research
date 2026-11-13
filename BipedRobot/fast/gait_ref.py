"""A hand-written walking pattern for the V2 robot, for a policy that learns corrections on top of it (validity.gait_ref in fast/valid_env.py).

One gait cycle per period_s: the right foot is on the floor for duty of the cycle, centred on a quarter cycle, and the left foot half a cycle later, so
with period 1 s and duty 0.6 there are 2 steps a second and 10% double support at each change of foot. Both soles stay level (the sole's flat face from the
foot mesh, ValidLocoEnv.sole, parallel to the floor with the torso upright) on the floor; in the air the swing foot may pitch freely
mid-swing (its pitch is released on a sin^2 weight, so it leaves and lands level), because this ankle bends only 19 deg past standing toward the shin and a
level foot cannot be lifted far. The stance foot slides back under the hip at speed_m_s. The swing foot leaves
and lands with that same speed, so it does not scuff or slap, rises on a sin^2 bump to lift_m, and moves forward on a cubic. The torso sways over the stance
foot by sway_m. crouch_m lowers the hips, which leaves the knees room to reach.

Joint angles come from damped least-squares inverse kinematics on the robot model with the torso fixed upright, solved for the right leg only. The left leg
is the right leg half a cycle later, which is exactly what the mirror augmentation assumes (fast/mirror.py: V2's mirrored joint angles carry sign +1).
The model is not exactly symmetric (feet sit 34.1 and 29.7 mm off the centre line), so the left foot lands a few mm off its mirror image; the policy's
corrections take that up.

Usage: python fast/gait_ref.py RESOLVED_OR_TRAIN_CONFIG OUT.npz [key=value ...]   (keys: period_s, duty, speed_m_s, lift_m, sway_m, crouch_m, samples)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import mujoco
import numpy as np

DEFAULTS = dict(period_s=1.0, duty=0.6, speed_m_s=0.2, lift_m=0.03, sway_m=0.012, crouch_m=0.01, samples=50, heel_deg=0.0, toe_deg=0.0, heel_frac=0.15, toe_start=0.6, stance_width_m=0.0)
# heel_deg, toe_deg: a human foot roll. The foot lands heel first with its toe up heel_deg, rolls flat by heel_frac of stance, and from toe_start of
# stance pivots about its toe until the heel is toe_deg up at toe-off. 0 and 0 keep the whole sole flat through stance.
# stance_width_m: above 0, the feet sit this far apart side to side (foot centres); 0 keeps the standing spacing, about 104 mm. 8 Nov 2026.


def right_foot_path(phase: np.ndarray, p: dict) -> np.ndarray:
    """Right foot offset from its standing place, in the torso frame (x forward, y left, z up), at each phase in [0, 1). Rows [dx, dy, dz]."""
    T, D, v = p["period_s"], p["duty"], p["speed_m_s"]
    X = v * T * D / 2.0  # the stance foot goes from +X to -X
    Ts = T * (1.0 - D)
    out = np.zeros((len(phase), 3))
    for i, ph in enumerate(np.asarray(phase) % 1.0):
        s_st = ((ph - (0.25 - D / 2.0)) % 1.0) / D  # 0..1 through stance, above 1 in swing
        if s_st <= 1.0:
            out[i, 0] = X - 2.0 * X * s_st
        else:
            s = (s_st - 1.0) * D / (1.0 - D)  # 0..1 through swing
            h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            out[i, 0] = h00 * (-X) + h10 * (-v * Ts) + h01 * X + h11 * (-v * Ts)  # leaves and lands at the floor's speed under the hip
            out[i, 2] = p["lift_m"] * np.sin(np.pi * s) ** 2
        out[i, 1] = p["sway_m"] * np.sin(2.0 * np.pi * ph)  # the torso is over the right foot at a quarter cycle, so the feet sit to its left
        out[i, 2] += p["crouch_m"]
    return out


def _level(R_foot: np.ndarray, ns: np.ndarray) -> np.ndarray:
    """The smallest rotation of a foot orientation that lays its sole face flat (its outward normal straight down)."""
    a = R_foot @ ns
    b = np.array([0.0, 0.0, -1.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return R_foot.copy()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K / (1.0 + c)) @ R_foot


def build_table(env, **params) -> dict:
    """Joint angles for one gait cycle, rows in phase order, columns in env joint order. Also reports the IK fit and the joint speeds it asks for."""
    p = {**DEFAULTS, **params}
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[0:3] = [0.0, 0.0, 0.5]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qa, da = np.asarray(env.joint_qpos_adrs), np.asarray(env.joint_dof_adrs)
    d.qpos[qa] = env.default_qpos
    mujoco.mj_kinematics(m, d)
    g = env.foot_geom_ids[0]
    body = int(m.geom_bodyid[g])
    ns = env.sole()[0][0]
    torso = d.xpos[env.torso_body_id].copy()
    home = d.geom_xpos[g].copy() - torso
    R_goal = _level(d.geom_xmat[g].reshape(3, 3).copy(), ns)
    lo, hi = env.jnt_min[:6], env.jnt_max[:6]
    phase = np.arange(int(p["samples"])) / int(p["samples"])
    path = right_foot_path(phase, p)
    if p["stance_width_m"] > 0:
        path[:, 1] += (abs(float(d.geom_xpos[env.foot_geom_ids[1], 1] - d.geom_xpos[g, 1])) - p["stance_width_m"]) / 2.0  # the right foot is at -y, so inward is +y
    _, on, f_loc, _ = env.sole()[0]
    a = on @ f_loc
    heel_loc, toe_loc = on[a < a.min() + 1e-3].mean(axis=0), on[a > a.max() - 1e-3].mean(axis=0)

    def rot_y(t):
        c, s_ = np.cos(t), np.sin(t)
        return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])

    def roll_offset(phi, pivot_loc):
        """Shift of the foot's origin when it pitches phi (toe up positive) about a sole point that stays put, and its new orientation."""
        Rot = rot_y(-phi)
        return (Rot - np.eye(3)) @ (-(R_goal @ pivot_loc)), Rot @ R_goal

    smooth = lambda x: 3.0 * np.clip(x, 0, 1) ** 2 - 2.0 * np.clip(x, 0, 1) ** 3
    hs, to = np.radians(p["heel_deg"]), np.radians(p["toe_deg"])
    q = env.default_qpos[:6].astype(np.float64).copy()
    side = np.array([0.0, 1.0, 0.0])
    D = p["duty"]
    s_st = ((phase - (0.25 - D / 2.0)) % 1.0) / D
    free = np.where(s_st > 1.0, np.sin(np.pi * np.clip((s_st - 1.0) * D / (1.0 - D), 0.0, 1.0)) ** 2, 0.0)
    verts = env._foot_verts[0] if getattr(env, "_foot_verts", None) is not None else (env.foot_clearance(), env._foot_verts[0])[1]
    lowest, pitch = [], []
    right, pos_err, rot_err = [], [], []
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for k in range(len(phase)):
        target = torso + home + path[k]
        if s_st[k] <= 1.0:
            u = s_st[k]
            if u < p["heel_frac"] and hs > 0:
                off, R_goal_k = roll_offset(hs * (1.0 - smooth(u / p["heel_frac"])), heel_loc)
            elif u > p["toe_start"] and to > 0:
                off, R_goal_k = roll_offset(-to * smooth((u - p["toe_start"]) / (1.0 - p["toe_start"])), toe_loc)
            else:
                off, R_goal_k = np.zeros(3), R_goal
        else:
            b = smooth((s_st[k] - 1.0) * D / (1.0 - D))
            off_to, _ = roll_offset(-to, toe_loc)
            off_hs, _ = roll_offset(hs, heel_loc)
            off = (1.0 - b) * off_to + b * off_hs
            R_goal_k = rot_y(-((1.0 - b) * (-to) + b * hs)) @ R_goal
        target = target + off
        for _ in range(200):
            d.qpos[qa[:6]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.geom_xmat[g].reshape(3, 3)
            e_p = target - d.geom_xpos[g]
            e_r = 0.5 * sum(np.cross(R[:, i], R_goal_k[:, i]) for i in range(3))
            e_r = e_r - free[k] * (e_r @ side) * side  # mid-swing, the foot's pitch is free
            e = np.r_[e_p, e_r]
            if np.linalg.norm(e_p) < 1e-6 and np.linalg.norm(e_r) < 1e-5:
                break
            mujoco.mj_jac(m, d, jp, jr, d.geom_xpos[g], body)
            J = np.r_[jp[:, da[:6]], jr[:, da[:6]]]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e), lo, hi)
        right.append(q.copy())
        lowest.append(float(np.min(d.geom_xpos[g, 2] + verts @ d.geom_xmat[g].reshape(3, 3)[2, :])))
        pitch.append(float(np.degrees(np.arcsin(np.clip(-(d.geom_xmat[g].reshape(3, 3) @ ns) @ np.array([1.0, 0.0, 0.0]), -1, 1)))))
        pos_err.append(float(np.linalg.norm(e_p)))
        rot_err.append(float(np.degrees(np.linalg.norm(e_r))))
    right = np.array(right)
    lowest = np.array(lowest)
    stance = s_st <= 1.0
    floor = float(np.median(lowest[stance]))
    stance_err = max(e for e, st in zip(rot_err, stance) if st)
    half = len(phase) // 2
    table = np.c_[right, np.roll(right, -half, axis=0)]  # left at phase k = right at phase k + half a cycle
    speed = np.abs(np.diff(np.r_[table, table[:1]], axis=0)) / (p["period_s"] / len(phase))
    return dict(table=table.astype(np.float32), phase=phase, params=p, foot_path=path, ik_pos_err_max_m=max(pos_err), ik_rot_err_max_deg=max(rot_err),
                joint_speed_max_rad_s=float(speed.max()), joint_speed_max_by_joint=speed.max(axis=0), at_limit=bool(np.any(np.isclose(right, lo) | np.isclose(right, hi))),
                stance_rot_err_max_deg=stance_err, stance_level_spread_mm=1000 * float(np.ptp(lowest[stance])), swing_clearance_max_mm=1000 * float(lowest.max() - floor),
                swing_pitch_deg=np.array(pitch))  # the right foot's pitch at each phase: positive heel up (toe down), negative toe up


def lookup(table: np.ndarray, phase: float) -> np.ndarray:
    """Joint angles at any phase, linear between samples, wrapping at the end of the cycle."""
    n = len(table)
    x = (phase % 1.0) * n
    i = int(np.floor(x)) % n
    f = x - np.floor(x)
    return (1.0 - f) * table[i] + f * table[(i + 1) % n]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import yaml
    from BipedRobot.fast.budget_env import BudgetLocoEnv
    from BipedRobot.fast.valid_env import ValidLocoEnv
    from BipedRobot.train_her_biped import build_env_config
    cfg_path, out = sys.argv[1], sys.argv[2]
    params = {k: float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
    c = yaml.safe_load(open(cfg_path))
    ec = build_env_config(c, "easy", False) if "env" in c else copy.deepcopy(c)
    for s in ("validity", "token_budget"):
        if s in c:
            ec[s] = copy.deepcopy(c[s])
    ec.get("validity", {}).pop("gait_ref", None)
    env = (BudgetLocoEnv if ec.get("token_budget") else ValidLocoEnv)(config=ec)
    env.reset(seed=0)
    r = build_table(env, **params)
    np.savez(out, table=r["table"], **{k: np.float64(v) for k, v in r["params"].items()})
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k in ("ik_pos_err_max_m", "ik_rot_err_max_deg", "joint_speed_max_rad_s", "at_limit")})
    print("speed by joint", np.round(r["joint_speed_max_by_joint"][:6], 2).tolist())
    print("range by joint (deg)", np.round(np.degrees(np.ptp(r["table"][:, :6], axis=0)), 1).tolist())
    env.close()


if __name__ == "__main__":
    main()
