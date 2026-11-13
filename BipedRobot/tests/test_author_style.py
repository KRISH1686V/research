"""The relative tracking kernel (reward.track_rel) and the clock-timed swing height (shaping.clock_height). Run by path from the workspace root."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.envs.rewards.shared_reward import tracking_reward_batch  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv, decode_rate  # noqa: E402
from BipedRobot.fast.replay import RewardConfig, relabel_reward  # noqa: E402
from BipedRobot.tests.test_budget_her import env_config, fill, make_env  # noqa: E402
from BipedRobot.tests.test_budget_window import brute_window  # noqa: E402
from BipedRobot.tests.test_step_budget import STEPS, rollout  # noqa: E402

REL = {"v_rel": 0.7, "v_min": 0.07, "w_rel": 0.7, "w_min": 0.15}
CLOCK = {"clock_height": {"weight": 0.3, "height_m": 0.025, "std_m": 0.012}}


def build(rel=True, shaping=None, validity=None, **budget):
    ec = env_config(**budget)
    if rel:
        ec["reward"] = {**ec.get("reward", {}), "track_rel": REL}
    ec["shaping"] = {**ec.get("shaping", {}), **(shaping or {})}
    ec["validity"] = {**ec["validity"], **(validity or {})}
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env


def test_kernel_values():
    env = build()
    k = lambda a, g: float(env.relative_track(np.array([a], np.float32), np.array([g], np.float32))[0])
    assert abs(k([0, 0, 0], [0.2, 0, 0]) - np.exp(-0.04 / 0.14 ** 2)) < 1e-6, "standing at 0.2 m/s"
    assert abs(k([0, 0, 0], [0.2, 0, 0]) - 0.1300) < 1e-3
    assert abs(k([0.05, 0, 0], [0, 0, 0]) - np.exp(-0.0025 / 0.07 ** 2)) < 1e-6, "the width never drops below v_min"
    assert abs(k([0.2, 0, 0.15], [0.2, 0, 0]) - np.exp(-1.0)) < 1e-6, "yaw width w_min at no turn command"
    assert abs(k([0.2, 0, 0.5], [0.2, 0, 0.4]) - np.exp(-0.01 / 0.28 ** 2)) < 1e-6, "yaw width follows the turn command"
    env.close()


def test_gpu_relabel_uses_the_same_kernel():
    env = build()
    cfg = RewardConfig.from_env(env)
    assert cfg.track_rel == (0.7, 0.07, 0.7, 0.15)
    rng = np.random.default_rng(0)
    a = rng.uniform([-0.1, -0.2, -0.5], [0.5, 0.2, 0.5], (500, 3)).astype(np.float32)
    g = rng.uniform([0.0, -0.2, -0.5], [0.5, 0.2, 0.5], (500, 3)).astype(np.float32)
    z = torch.zeros(500)
    got = relabel_reward(cfg, torch.as_tensor(a), torch.as_tensor(g), z, z, z, z, z).numpy()
    progress = np.where(g[:, 0] > 0.1, np.clip(a[:, 0] / np.maximum(g[:, 0], 1e-9), 0, 1), 0.0) * env.progress_weight
    assert np.allclose(got, env.relative_track(a, g) + progress, atol=1e-5)
    env.close()


def test_off_by_default():
    env = make_env()
    assert env.track_rel is None and env.clock_height_weight == 0.0 and RewardConfig.from_env(env).track_rel == ()
    env.close()


def test_env_reward_swaps_only_the_tracking_term():
    plain, rel = build(rel=False), build()
    goal = np.array([0.2, 0, 0, 1, 1, 1], np.float32)
    o1, _ = plain.reset(seed=3, options={"desired_goal": goal})
    o2, _ = rel.reset(seed=3, options={"desired_goal": goal})
    rng = np.random.default_rng(2)
    swapped = 0
    for _ in range(400):
        act = rng.uniform(-0.5, 0.5, 12).astype(np.float32)
        o1, r1, t1, tr1, i1 = plain.step(act)
        o2, r2, t2, tr2, i2 = rel.step(act)
        assert np.array_equal(plain.data.qpos, rel.data.qpos), "the kernel must not change the physics"
        assert i1["budget_overdrawn"] == i2["budget_overdrawn"]
        a, g = o2["achieved_goal"][None], o2["desired_goal"][None, :3]
        if i2["fell"] or i2["budget_overdrawn"]:
            assert r1 == r2
        else:
            want = float(rel.relative_track(a, g)[0] - tracking_reward_batch(a, g, sigma_v=plain.sigma_v, sigma_w=plain.sigma_w)[0])
            assert abs((r2 - r1) - want) < 1e-5
            assert abs(i2["r_track"] - float(rel.relative_track(a, g)[0])) < 1e-6
            swapped += 1
        if t1 or tr1:
            o1, _ = plain.reset(options={"desired_goal": goal})
            o2, _ = rel.reset(options={"desired_goal": goal})
    assert swapped > 100
    plain.close()
    rel.close()


def test_relabeling_with_the_relative_kernel_matches_brute_force():
    W = 20
    budget = {**make_env.__globals__["BUDGET"], **STEPS, "window_steps": W, "alive_bonus": 0.2}
    env = build(**STEPS, window_steps=W, alive_bonus=0.2)
    data = rollout(env, 1500, 4)
    rb = fill(env, data, budget=budget)
    assert rb.cfg.track_rel == (0.7, 0.07, 0.7, 0.15)
    N = 4
    index, ep = {}, []
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            for t, e in enumerate(ep):
                index[np.concatenate((e["obs"][:3], e["obs"][6 + N:-N])).tobytes()] = (ep, t)
            ep = []
    checked = 0
    for _ in range(3):
        obs, _, rew, nxt, _, _ = [x.cpu().numpy() for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 6 + N:-N])).tobytes()]
            goal = obs[j, 3:6 + N].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = brute_window(np.array([e["info"]["budget_cost"] for e in ep], dtype=np.float64), rate, W)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            want = base if e["info"]["fell"] else (0.0 if (deficits[t + 1] > cap).any() else base) + env.alive_bonus
            assert abs(want - rew[j]) < 1e-5
            checked += 1
    assert checked > 700
    env.close()


def test_clock_height_needs_the_clock_and_gait_shaping():
    with pytest.raises(ValueError, match="clock_height"):
        build(shaping=CLOCK)
    with pytest.raises(ValueError, match="clock_height"):
        build(shaping=CLOCK, validity={"clock_hz": 1.0})


def test_clock_target_shape():
    env = build(shaping={**CLOCK, "gait_shaping": True}, validity={"clock_hz": 1.0})
    for p, want in ((0.0, [0, 0]), (0.25, [0, 0.025]), (0.5, [0, 0]), (0.75, [0.025, 0]), (0.125, [0, 0.025 * np.sin(np.pi / 4)])):
        env._phase = p
        assert np.allclose(env.clock_swing_target(), want, atol=1e-9), p
    env.close()


def test_clock_height_adds_its_term_and_nothing_else():
    shaping = {"gait_shaping": True}
    base, clock = build(shaping=shaping, validity={"clock_hz": 1.0}), build(shaping={**shaping, **CLOCK}, validity={"clock_hz": 1.0})
    goal = np.array([0.2, 0, 0, 1, 1, 1], np.float32)
    base.reset(seed=5, options={"desired_goal": goal})
    clock.reset(seed=5, options={"desired_goal": goal})
    rng = np.random.default_rng(4)
    best = 0.0
    for t in range(1, 300):
        act = rng.uniform(-0.6, 0.6, 12).astype(np.float32)
        _, _, term, trunc, i1 = base.step(act)
        _, _, _, _, i2 = clock.step(act)
        assert np.array_equal(base.data.qpos, clock.data.qpos)
        p = (t * clock.dt * 1.0) % 1.0
        h = 0.025 * np.sin(2 * np.pi * (p % 0.5))
        target = np.array([0.0, h]) if p < 0.5 else np.array([h, 0.0])
        match = np.exp(-np.sum((clock.foot_clearance() - target) ** 2) / 0.012 ** 2)
        assert abs(i2["r_gait_move"] - i1["r_gait_move"] - 0.3 * match) < 1e-6, t
        assert i2["r_gait_still"] == i1["r_gait_still"] and i2["r_flight"] == i1["r_flight"]
        best = max(best, match)
        if term or trunc:
            break
    assert best > 0.3, "the test must see a foot near the target"
    base.close()
    clock.close()


FLAT = {"foot_flat": {"l1": 1.0, "l2": 2.0}}


def test_foot_flat_needs_smooth_penalties():
    with pytest.raises(ValueError, match="foot_flat"):
        build(shaping={**FLAT, "smooth_penalties": False})


def test_foot_flat_charges_tilted_feet_on_the_floor_only():
    plain, flat = build(), build(shaping=FLAT)
    goal = np.array([0.2, 0, 0, 1, 1, 1], np.float32)
    plain.reset(seed=8, options={"desired_goal": goal})
    flat.reset(seed=8, options={"desired_goal": goal})
    for _ in range(20):
        _, _, _, _, i2 = flat.step(np.zeros(12, np.float32))
        plain.step(np.zeros(12, np.float32))
    assert i2["foot_flat_cost"] < 0.1, "standing on flat feet costs almost nothing"
    rng = np.random.default_rng(9)
    charged = 0
    for _ in range(300):
        act = rng.uniform(-0.6, 0.6, 12).astype(np.float32)
        _, _, term, trunc, i1 = plain.step(act)
        _, _, _, _, i2 = flat.step(act)
        assert np.array_equal(plain.data.qpos, flat.data.qpos)
        cost = 0.0
        for k, g in enumerate(flat.foot_geom_ids):
            if flat._feet_contact()[k] > 0.5:
                down = flat.data.geom_xmat[g].reshape(3, 3) @ flat.sole()[k][0]
                t = np.arccos(np.clip(-down[2], -1, 1))  # angle of the sole's face to the floor
                cost += abs(np.sin(t)) + 2.0 * np.sin(t) ** 2
        assert abs(i2["foot_flat_cost"] - cost) < 1e-6
        ramp = max(0.0, min(1.0, flat.smoothness_ramp))
        assert abs((i1["r_smooth"] - i2["r_smooth"]) - ramp * cost) < 1e-6
        assert i2["r_gait_move"] == i1["r_gait_move"]
        charged += int(cost > 0.05)
        if term or trunc:
            break
    assert charged > 10
    plain.close()
    flat.close()


def test_sole_reference_is_the_reset_pose_with_or_without_budgets():
    flat = build(shaping=FLAT, costs=["flight", "slip", "torque", "tiltstance"], rate_min=[0.002, 0.02, 0.05, 0.02], rate_max=[0.2, 0.2, 0.8, 2.0])
    goal = np.array([0.2, 0, 0, 1, 1, 1, 1], np.float32)
    for seed in (1, 2):
        flat.reset(seed=seed, options={"desired_goal": goal})
        assert np.all(flat.sole_tilt() < 1e-9), "the reference is the reset pose"
        for _ in range(30):
            _, _, term, _, info = flat.step(np.random.default_rng(seed).uniform(-0.5, 0.5, 12).astype(np.float32))
            if term:
                break
        want = float(np.sum((flat._feet_contact() > 0.5) & (flat.sole_tilt() > flat.tilt_rad)))
        assert info["budget_cost"][3] == np.float32(want)
    flat.close()


SOLE = dict(costs=["flight", "slip", "torque", "partsole"], rate_min=[0.002, 0.02, 0.05, 0.02], rate_max=[0.2, 0.2, 0.8, 2.0])


def v2_env(name="v2_deploy_budget_alpha30_human_rel_sole"):
    import yaml
    from BipedRobot.train_her_biped import build_env_config
    c = yaml.safe_load(open(Path(__file__).resolve().parents[1] / f"config/{name}.yaml"))
    ec = build_env_config(c, "easy", False)
    ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
    ec.setdefault("shaping", {})["reset_velocity"] = False
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env


def corner_gap(env, k):
    """Independent of sole_gap: the sole is a rectangle, so its corners' height spread is length * |slope along| + width * |slope across|."""
    ns, on, f, l = env.sole()[k]
    R = env.data.geom_xmat[env.foot_geom_ids[k]].reshape(3, 3)
    return np.ptp(on @ f) * abs((R @ f)[2]) + np.ptp(on @ l) * abs((R @ l)[2])


def test_v2_sole_is_the_flat_bottom_face():
    env = v2_env()
    env.reset(seed=1, options={"desired_goal": np.array([0.2, 0, 0, 0.5, 0.5, 0.5, 1, 0, 1], np.float32)})
    for k, g in enumerate(env.foot_geom_ids):
        ns, on, f, l = env.sole()[k]
        assert len(on) == 4, "the V2 sole face is a rectangle"
        assert 0.085 < np.ptp(on @ f) < 0.095 and 0.033 < np.ptp(on @ l) < 0.042, (np.ptp(on @ f), np.ptp(on @ l))
        R = env.data.geom_xmat[g].reshape(3, 3)
        assert (R @ f)[0] > 0.95, "the length axis points forward at reset"
        assert abs(f @ ns) < 1e-9 and abs(l @ ns) < 1e-9 and abs(f @ l) < 1e-9
    assert np.degrees(env.sole_angle()).max() < 2.5, "the reset pose is within a couple of degrees of flat"
    for _ in range(50):
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert np.all(env.sole_gap() < 0.0005) and np.degrees(env.sole_angle()).max() < 0.3, (env.sole_gap(), np.degrees(env.sole_angle()))
    assert info["budget_cost"][env.cost_names.index("partsole")] == 0.0, "standing flat is not a part sole"
    env.close()


def test_part_sole_cost_heel_sign_and_gap():
    env = v2_env()
    goal = np.array([0.2, 0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    env.reset(seed=4, options={"desired_goal": goal})
    i = env.cost_names.index("partsole")
    rng = np.random.default_rng(12)
    seen = heel = 0
    for _ in range(500):
        _, _, term, trunc, info = env.step(rng.uniform(-0.7, 0.7, 12).astype(np.float32))
        want = 0
        for k, g in enumerate(env.foot_geom_ids):
            assert abs(env.sole_gap()[k] - corner_gap(env, k)) < 2e-4
            want += int(env._feet_contact()[k] > 0.5 and corner_gap(env, k) > 0.002)
            along = env.sole_slope()[k, 0]
            if along < -0.05:  # toe raised: the lowest corner must be at the heel end
                ns, on, f, _ = env.sole()[k]
                R = env.data.geom_xmat[g].reshape(3, 3)
                low = on[np.argmin(on @ R[2, :])]
                assert (low - on.mean(axis=0)) @ f < 0
                heel += 1
        if abs(env.sole_gap()[0] - 0.002) > 2e-4 and abs(env.sole_gap()[1] - 0.002) > 2e-4:
            assert info["budget_cost"][i] == np.float32(want)
        seen += want
        if term or trunc:
            env.reset(options={"desired_goal": goal})
    assert seen > 20 and heel > 5, (seen, heel)
    env.close()


def test_roll_only_foot_flat_and_foot_drag():
    shaping = {"gait_shaping": True}
    base = build(shaping=shaping)
    roll = build(shaping={**shaping, "foot_flat": {"l1": 1.0, "l2": 2.0, "axes": "roll"}, "foot_drag": {"weight": 0.75, "height_mm": 2}})
    with pytest.raises(ValueError, match="axes"):
        build(shaping={"foot_flat": {"l1": 1.0, "axes": "pitch"}})
    with pytest.raises(ValueError, match="foot_drag"):
        build(shaping={"foot_drag": {"weight": 0.75}})
    goal = np.array([0.2, 0, 0, 1, 1, 1], np.float32)
    base.reset(seed=6, options={"desired_goal": goal})
    roll.reset(seed=6, options={"desired_goal": goal})
    rng = np.random.default_rng(13)
    dragged = 0
    for _ in range(300):
        act = rng.uniform(-0.6, 0.6, 12).astype(np.float32)
        _, _, term, trunc, i1 = base.step(act)
        _, _, _, _, i2 = roll.step(act)
        assert np.array_equal(base.data.qpos, roll.data.qpos)
        down = roll._feet_contact() > 0.5
        across = roll.sole_slope()[:, 1]
        want_flat = float(np.sum(np.abs(across) * down) + 2.0 * np.sum((across * down) ** 2))
        assert abs(i2["foot_flat_cost"] - want_flat) < 1e-6
        ramp = max(0.0, min(1.0, roll.smoothness_ramp))
        assert abs((i1["r_smooth"] - i2["r_smooth"]) - ramp * want_flat) < 1e-6
        clear = roll.foot_clearance()
        want_drag = 0.0
        for k, g in enumerate(roll.foot_geom_ids):
            if not down[k]:
                v = np.zeros(6)
                mujoco.mj_objectVelocity(roll.model, roll.data, mujoco.mjtObj.mjOBJ_GEOM, g, v, 0)
                want_drag += (1 - np.clip(clear[k] / 0.002, 0, 1)) * np.hypot(v[3], v[4])
        assert abs(i2["foot_drag"] - want_drag) < 1e-9
        assert abs((i1["r_gait_move"] - i2["r_gait_move"]) - 0.75 * want_drag) < 1e-6
        dragged += int(want_drag > 0.01)
        if term or trunc:
            break
    assert dragged > 3
    base.close()
    roll.close()
