"""Options added 9 Nov 2026 from the designer's training setup: shaping.ref_full, validity.obs_vibration, the exactly left-right symmetric actor and its run_policy.py path. Each is off unless named. CPU only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2/hardware_tests"))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.fast.mirror import action_perm, obs_maps  # noqa: E402
from BipedRobot.fast.nets import Actor, actor_from_checkpoint  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "config/v2_deploy_budget_alpha30_r8ref_fit.yaml"
FULL = dict(table="config/gait_ref_r10author.npz", weights=dict(joint_pos=0.6, foot_pos=0.4, contact=0.5, orientation=0.2, lin_vel=0.3, ang_vel=0.2, joint_vel=0.1, foot_height=1.5),
            std=dict(joint_pos=0.15, joint_vel=1.5, foot_pos=0.02, orientation=0.08, lin_vel=0.1, ang_vel=0.3, foot_height=0.010), rsi_prob=1.0, rsi_joint_noise=0.0)


def make(full=True, validity=None, realism="medium"):
    c = yaml.safe_load(BASE.read_text())
    if full:
        c["shaping"].pop("ref_imitate")
        c["shaping"]["ref_full"] = dict(FULL)
    c["validity"].update(validity or {})
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        if k in c:
            ec[k] = c[k]
    return env_class(ec)(config=ec, render_mode=None)


def maps():
    return obs_maps(5, 3, True, True)


def actor_and_obs(mirror=True):
    perm, sign = maps()
    keep = list(range(3, len(perm) - 3))
    m = dict(obs_perm=perm, obs_sign=sign, act_perm=action_perm()) if mirror else None
    torch.manual_seed(0)
    return Actor(len(perm), 12, keep=keep, mirror=m).eval(), torch.randn(16, len(perm)), perm, sign, keep


def test_actor_is_exactly_mirror_symmetric():
    a, obs, perm, sign, _ = actor_and_obs()
    mirrored = obs[:, perm] * torch.as_tensor(sign, dtype=torch.float32)
    assert torch.allclose(a.deterministic(obs), a.deterministic(mirrored)[:, action_perm()], atol=1e-6)
    mean, log_std = a.dist_params(obs)
    mean_m, log_std_m = a.dist_params(mirrored)
    assert torch.allclose(log_std, log_std_m[:, action_perm()], atol=1e-5) and torch.allclose(mean, mean_m[:, action_perm()], atol=1e-5)


def test_mirror_applied_twice_is_the_identity():
    perm, sign = maps()
    x = np.arange(len(perm), dtype=np.float64) + 1.0
    once = x[perm] * sign
    assert np.allclose(once[perm] * sign, x)


def test_plain_actor_is_unchanged_and_not_symmetric():
    plain, obs, perm, sign, keep = actor_and_obs(mirror=False)
    assert "obs_perm" not in plain.state_dict()
    a, b = plain.deterministic(obs), plain.deterministic(obs[:, perm] * torch.as_tensor(sign, dtype=torch.float32))[:, action_perm()]
    assert not torch.allclose(a, b, atol=1e-3)
    h = plain.latent_pi(obs[:, keep])
    assert torch.allclose(plain.dist_params(obs)[0], plain.mu(h))


def test_checkpoint_keeps_the_mirror():
    a, obs, *_ = actor_and_obs()
    back = actor_from_checkpoint(dict(actor=a.state_dict(), obs_dim=obs.shape[1], act_dim=12))
    assert torch.allclose(a.deterministic(obs), back.deterministic(obs))
    plain, obs2, *_ = actor_and_obs(mirror=False)
    assert "obs_perm" not in actor_from_checkpoint(dict(actor=plain.state_dict(), obs_dim=obs2.shape[1], act_dim=12)).state_dict()


def test_robot_code_matches_the_trained_actor(tmp_path):
    import run_policy
    a, obs, perm, sign, keep = actor_and_obs()
    sd = {k: v.numpy() for k, v in a.state_dict().items()}
    f = tmp_path / "p.npz"
    np.savez(f, w0=sd["latent_pi.0.weight"], b0=sd["latent_pi.0.bias"], w1=sd["latent_pi.2.weight"], b1=sd["latent_pi.2.bias"], mu_w=sd["mu.weight"], mu_b=sd["mu.bias"], keep=np.array(keep),
             obs_dim=len(perm), history=5, reads_truth=False, scale=np.ones(5, np.float32), default=np.zeros(12, np.float32), lo=-np.ones(12, np.float32), hi=np.ones(12, np.float32),
             ctrl_min=-np.ones(12, np.float32), ctrl_max=np.ones(12, np.float32), fraction=0.9, alpha=0.2, servo_names=np.array([str(i) for i in range(12)]), meta=json.dumps({}), clock_hz=1.0,
             mirror_obs_perm=perm, mirror_obs_sign=sign, mirror_act_perm=action_perm())
    core = run_policy.PolicyCore(str(f))
    for row in obs[:4].numpy():
        assert np.allclose(core.policy_action(row), a.deterministic(torch.as_tensor(row)[None])[0].detach().numpy(), atol=1e-5)


def test_full_reference_pays_about_one_on_the_table_and_less_off_it():
    env = make()
    env.reset(seed=3)  # rsi_prob 1: starts on the table
    t = env.ref_full_terms()
    assert t["joint_pos"] > 0.999 and t["foot_pos"] > 0.99 and t["orientation"] > 0.95
    for j, q in zip(env.joint_qpos_adrs, env.data.qpos[env.joint_qpos_adrs] + 0.2):
        env.data.qpos[j] = q
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    off = env.ref_full_terms()
    assert off["joint_pos"] < 0.3 and off["foot_pos"] < t["foot_pos"]
    env.close()


def test_foot_height_and_contact_take_the_worse_foot():
    env = make()
    env.reset(seed=3)
    h = np.array([0.0, 0.0])
    env.foot_clearance = lambda: h
    env._feet_contact = lambda: np.array([1.0, 1.0])
    env._rf = dict(env._rf, foot_height=np.tile([0.02, 0.0], (50, 1)), contact=np.tile([0.0, 1.0], (50, 1)))
    t = env.ref_full_terms()  # right should be 20 mm up and is on the floor; left is fine
    assert t["foot_height"] == pytest.approx(np.exp(-4.0)) and t["contact"] == pytest.approx(0.0)
    env.close()


def test_reference_reward_replaces_ref_imitate_and_goes_in_the_gait_term():
    env = make()
    obs, _ = env.reset(seed=5)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert "ref_full" in info and "ref_match" not in info and info["ref_full"] > 0.0
    env.close()
    with pytest.raises(ValueError):
        c = yaml.safe_load(BASE.read_text())
        c["shaping"]["ref_full"] = dict(FULL)
        ec = build_env_config(c, "medium", False)
        for k in ("validity", "token_budget"):
            if k in c:
                ec[k] = c[k]
        env_class(ec)(config=ec, render_mode=None)


def test_off_by_default():
    env = make(full=False)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert "ref_full" not in info and env._rf is None and env._vib is None and not env._vib_cfg
    env.close()


def test_vibration_only_in_training_and_inside_its_band():
    v = dict(obs_vibration=dict(gyro=1.2, joint_vel=0.8, hz=[3.0, 20.0], amp=[1.0, 1.0]))
    env = make(validity=v, realism="medium")
    env.reset(seed=1)
    assert env._vib is None
    env.close()
    env = make(validity=v, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=1)
    g, j = env._vib
    assert g.shape[1] == 3 and j.shape[1] == 12
    assert np.allclose(g.std(axis=0), 1.2, atol=1e-3) and np.allclose(j.std(axis=0), 0.8, atol=1e-3)
    p = np.abs(np.fft.rfft(g, axis=0)) ** 2
    f = np.fft.rfftfreq(len(g), env.dt)
    assert p[(f < 3.0) | (f > 20.0)].sum() < 1e-6 * p.sum()
    env.set_difficulty(0.0)
    env.reset(seed=2)
    assert np.allclose(env._vib[0], 0.0)
    env.close()


def test_vibration_reaches_the_policy_but_not_the_critic():
    v = dict(obs_vibration=dict(gyro=1.2, joint_vel=0.8, hz=[3.0, 20.0], amp=[1.0, 1.0]))
    env = make(validity=v, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=1)
    o, _, _, _, _ = env.step(np.zeros(12, np.float32))
    n, H = env.num_joints, env.history
    seen = o["observation"][42 * (H - 1):42 * H]
    assert abs(float(seen[2 * n] - env._priv[2 * n])) > 1e-4  # gyro x differs
    env.close()
