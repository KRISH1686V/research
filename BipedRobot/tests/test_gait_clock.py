"""Gait clock and phase reward, heading hold, and the Pi runner with a clock. Run by path from the workspace root."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(REPO / "diagnostics/audit_2026-10-31/audit/v2/hardware_tests"))
import export_policy  # noqa: E402
import run_policy as rp  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.mirror import action_perm, obs_maps  # noqa: E402
from BipedRobot.fast.nets import Actor, actor_inputs  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOAL = np.array([0.2, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)


def config(name="v2_deploy_budget_gait", realism="easy", **validity):
    c = yaml.safe_load(open(REPO / f"config/{name}.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"] = {"friction": True, "motor_torque": True, "history": 5, **validity}
    ec["token_budget"] = c["token_budget"]
    ec.setdefault("shaping", {})["reset_velocity"] = False
    return ec


def make(ec):
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env


def test_clock_values_and_layout():
    env = make(config(clock_hz=2.0))
    o, _ = env.reset(seed=1, options={"desired_goal": GOAL})
    assert o["observation"].shape == (215,), "5 frames of 42, sine and cosine, 3 balances"
    assert np.allclose(o["observation"][210:212], [0.0, 1.0], atol=1e-6)
    for k in range(1, 40):
        o, _, term, _, _ = env.step(np.zeros(12, np.float32))
        angle = 2 * math.pi * ((k * env.dt * 2.0) % 1.0)
        assert np.allclose(o["observation"][210:212], [math.sin(angle), math.cos(angle)], atol=1e-5), k
        if term:
            break
    o, _ = env.reset(seed=2, options={"desired_goal": GOAL})
    assert np.allclose(o["observation"][210:212], [0.0, 1.0], atol=1e-6), "a new episode restarts the clock"
    env.close()


def test_clock_off_by_default_keeps_the_old_layout():
    env = make(config())
    o, _ = env.reset(seed=1, options={"desired_goal": GOAL})
    assert o["observation"].shape == (213,) and env.clock_hz == 0.0 and env.phase_weight == 0.0
    env.close()


def test_mirror_flips_the_clock_and_keeps_balances_last():
    perm, sign = obs_maps(5, 3, True)
    dim = 3 + 6 + 210 + 2 + 3
    assert len(perm) == dim and sorted(perm) == list(range(dim))
    assert list(perm[219:221]) == [219, 220] and list(sign[219:221]) == [-1.0, -1.0]
    assert list(perm[221:]) == [221, 222, 223] and list(sign[221:]) == [1.0, 1.0, 1.0]
    same, same_sign = obs_maps(5, 3)
    assert np.array_equal(perm[:219], same[:219]) and np.array_equal(sign[:219], same_sign[:219]), "everything before the clock stays where it was"


def test_phase_reward_needs_the_clock_and_gait_shaping():
    ec = config()
    ec["shaping"]["phase_weight"] = 0.5
    with pytest.raises(ValueError, match="phase_weight"):
        make(ec)


def test_phase_reward_is_exact_and_the_foot_order_is_right():
    ec = config(clock_hz=2.0)
    ec["shaping"]["phase_weight"] = 0.5
    env = make(ec)
    plain_cfg = config(clock_hz=2.0)
    plain = make(plain_cfg)
    env.reset(seed=1, options={"desired_goal": GOAL})
    plain.reset(seed=1, options={"desired_goal": GOAL})
    rng = np.random.default_rng(0)
    seen = set()
    for k in range(1, 80):
        act = rng.uniform(-0.2, 0.2, 12).astype(np.float32)
        _, _, term, _, a = env.step(act)
        _, _, _, _, b = plain.step(act)
        phase = (k * env.dt * 2.0) % 1.0
        fn, _, _ = env.substep_loads()
        right, left = env._feet_contact() > 0.5
        first = phase < 0.5
        want = 0.5 * (int(right == first) + int(left != first) - 1)
        assert abs(a["r_gait_move"] - b["r_gait_move"] - want) < 1e-9
        seen.add(want)
        if term:
            break
    assert len(seen) >= 2, "the test must see the reward change"
    names = [env.model.body(i).name for i in env.foot_body_ids]
    assert names[0].startswith("r_") and names[1].startswith("l_"), names
    env.close()
    plain.close()


def test_relabeling_keeps_the_phase_reward():
    budget = {"rate_min": [10.0] * 3, "rate_max": [20.0] * 3, "burst_steps": 100, "p_keep": 1.0, "p_achieved": 0.0, "alive_bonus": 0.2}
    ec = config(clock_hz=2.0)
    ec["token_budget"] = {**ec["token_budget"], **budget, "p_unconstrained": 1.0, "window_steps": None}
    ec["shaping"]["phase_weight"] = 0.5
    env = make(ec)
    dim = sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())
    rb = BudgetHerReplay(3000, 1, 256, 2, dim, 12, RewardConfig.from_env(env), DEVICE, budget)
    rng = np.random.default_rng(0)
    o, _ = env.reset(seed=2, options={"desired_goal": GOAL})
    i = rb.i_rew
    stored = {}
    for _ in range(900):
        a = rng.uniform(-0.4, 0.4, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        row = np.zeros((1, rb.width), dtype=np.float32)
        row[0, :dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = flatten_obs(o), flatten_obs(o2), a
        row[0, i:i + 10] = (r, term, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([term or trunc]))
        stored[np.concatenate((flatten_obs(o)[:3], flatten_obs(o)[9:])).tobytes()] = (o2["achieved_goal"].copy(), info, r)
        o = o2
        if term or trunc:
            o, _ = env.reset(options={"desired_goal": GOAL})
    obs, _, rew, _, _, _ = [x.cpu().numpy() for x in rb.sample()]
    checked = 0
    for j in range(rb.nb_virtual):
        ach, info, _ = stored[np.concatenate((obs[j, :3], obs[j, 9:])).tobytes()]
        want = float(env.compute_reward(ach, obs[j, 3:6], info))
        want = want if info["fell"] else want + 0.2
        assert abs(want - rew[j]) < 1e-4
        checked += 1
    assert checked > 100
    env.close()


def exported(tmp_path, hz):
    ec = config(clock_hz=hz) if hz else config()
    dim = 222 + (2 if hz else 0)
    torch.manual_seed(0)
    actor = Actor(dim, 12, keep=actor_inputs(dim, 6, 3, {"achieved", "levels"})).eval()
    ckpt = tmp_path / "p.pt"
    torch.save({"actor": actor.state_dict(), "obs_dim": dim, "act_dim": 12, "steps": 7}, ckpt)
    rc = tmp_path / "r.yaml"
    rc.write_text(yaml.safe_dump({"resolved_env_config": ec}))
    out = tmp_path / "p.npz"
    export_policy.main([str(ckpt), str(rc), str(out)])
    return actor, ec, str(out)


@pytest.mark.parametrize("hz", [0.0, 2.0])
def test_pi_runner_matches_the_env_with_and_without_a_clock(tmp_path, hz):
    actor, ec, path = exported(tmp_path, hz)
    core = rp.PolicyCore(path)
    assert core.clock_hz == hz
    env = make(ec)
    o, _ = env.reset(seed=3, options={"desired_goal": GOAL})
    for _ in range(60):
        d = env.data
        s = (np.array([d.qpos[a] for a in env.joint_qpos_adrs], np.float32), np.array([d.qvel[a] for a in env.joint_dof_adrs], np.float32), d.qvel[3:6].astype(np.float32), env._get_projected_gravity())
        action, target = core.step(*s, GOAL[:3], GOAL[3:])
        want = actor.deterministic(torch.as_tensor(flatten_obs(o))[None])[0].detach().numpy()
        assert np.allclose(action, want, atol=2e-5)
        o, _, term, _, _ = env.step(action)
        assert np.allclose(target, env.data.ctrl[env.actuator_ids], atol=1e-5)
        if term:
            break
    env.close()


def test_heading_command():
    q = lambda yaw: (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    assert rp.heading_command(q(0.0), 0.0) == 0.0
    assert rp.heading_command(q(0.1), 0.0) == pytest.approx(-0.1)
    assert rp.heading_command(q(1.0), 0.0) == pytest.approx(-0.15), "limited to what training saw"
    assert rp.heading_command(q(-1.0), 0.0) == pytest.approx(0.15)
    assert rp.heading_command(q(3.1), -3.1) == pytest.approx(0.083185, abs=1e-4), "the short way across the +-pi seam"
