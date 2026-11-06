"""History, difficulty ramp, critic-only state, mirror and the averaged policy. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.mirror import action_perm, obs_maps, priv_maps  # noqa: E402
from BipedRobot.fast.replay import GpuHerReplay, RewardConfig  # noqa: E402
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cfg(realism="full", **val):
    c = yaml.safe_load(open(REPO / "config/v2_shaped.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"] = {**c["validity"], **val}
    return ec


def clean_cfg(realism="easy", **val):
    c = yaml.safe_load(open(REPO / "config/v2_shaped.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"] = dict(val)
    ec.setdefault("shaping", {})["reset_velocity"] = False
    return ec


def make(c):
    env = ValidLocoEnv(config=copy.deepcopy(c))
    env.reward_mode = "track"
    return env


def test_v2_model_is_loaded():
    env = make(cfg())
    assert env.xml_path.endswith("robot_v2_train.xml") and abs(float(env.model.body_mass.sum()) - 1.661) < 0.01
    env.close()


def test_history_is_oldest_first_and_refills_on_reset():
    env = make(clean_cfg(history=5))
    o, _ = env.reset(seed=1)
    assert o["observation"].shape == (210,)
    assert np.array_equal(o["observation"][:42], o["observation"][168:]), "a new episode fills all frames with the first one"
    prev = o["observation"].copy()
    o, *_ = env.step(np.ones(12, np.float32) * 0.1)
    assert np.array_equal(o["observation"][:168], prev[42:]), "frames must shift by one"
    env.reset(seed=2)
    assert len(env._hist) == 5
    env.close()


def test_difficulty_scales_and_restores():
    env = make(cfg())
    hard = dict(env._hard)
    env.set_difficulty(0.0)
    env.reset(seed=3)
    assert env.dr_friction_range == (0.8, 0.8) and env.dr_mass_scale_range == (1.0, 1.0) and env.dr_push_magnitude == 0.0 and env.backlash_deadband == 0.0
    assert env.delay_range == (0.025, 0.025)
    assert np.allclose(env._bias["joint_pos"], 0.0)
    assert abs(env.model.body_mass[env.torso_body_id] / env._nominal_torso_mass - 1.0) < 1e-6
    env.set_difficulty(0.5)
    env.reset(seed=3)
    assert np.allclose(env.dr_friction_range, (0.525, 0.8))
    env.set_difficulty(1.0)
    env.reset(seed=3)
    assert env.dr_friction_range == hard["friction"] and env.dr_push_magnitude == hard["push"] and env.delay_range == hard["delay"]
    env.dr_friction_range = (0.2, 0.2)  # test conditions set attributes directly at d = 1
    env.reset(seed=3)
    assert env.dr_friction_range == (0.2, 0.2)
    env.close()


def test_noise_off_at_zero_difficulty():
    a, b = make(clean_cfg(obs_noise={"joint_vel": 0.25, "gyro": 0.02})), make(clean_cfg())
    a.set_difficulty(0.0)
    oa, _ = a.reset(seed=4)
    ob, _ = b.reset(seed=4)
    assert np.allclose(oa["observation"], ob["observation"], atol=1e-6)
    a.close()
    b.close()


def test_privileged_state_is_clean_and_not_in_policy_input():
    noisy, clean = make(clean_cfg(obs_noise={"joint_vel": 0.25, "gyro": 0.02}, history=5)), make(clean_cfg())
    o, _ = noisy.reset(seed=5)
    c, _ = clean.reset(seed=5)
    p = noisy.privileged()
    assert p.shape == (46,) and np.allclose(p[:42], c["observation"], atol=1e-6)
    assert o["observation"].shape == (210,) and not np.allclose(o["observation"][-42:], c["observation"], atol=1e-4)
    noisy.close()
    clean.close()


def test_mirror_maps_are_involutions():
    for frames, budget in ((1, 0), (5, 0), (5, 3)):
        p, s = obs_maps(frames, budget)
        assert len(p) == 6 + budget + 42 * frames + budget and sorted(p) == list(range(len(p)))
        assert np.array_equal(p[p], np.arange(len(p))) and np.array_equal(s * s[p], np.ones(len(p)))
    a = action_perm()
    assert np.array_equal(a[a], np.arange(12))


def mirror_state(env, source):
    """Set env to the mirror image of `source`."""
    m, d, sd = env.model, env.data, source.data
    a = action_perm()
    q = sd.qpos.copy()
    v = sd.qvel.copy()
    q[1] = -q[1]
    w, x, y, z = q[3:7]
    q[3:7] = (w, -x, y, -z)
    v[1] = -v[1]
    v[3], v[5] = -v[3], -v[5]
    for i, adr in enumerate(env.joint_qpos_adrs):
        q[adr] = sd.qpos[source.joint_qpos_adrs[a[i]]]
    for i, adr in enumerate(env.joint_dof_adrs):
        v[adr] = sd.qvel[source.joint_dof_adrs[a[i]]]
    d.qpos[:] = q
    d.qvel[:] = v
    import mujoco
    mujoco.mj_forward(m, d)


def test_mirror_matches_the_simulator():
    a, b = make(clean_cfg()), make(clean_cfg())
    a.reset(seed=6, options={"desired_goal": np.array([0.2, 0.05, 0.1], np.float32)})
    b.reset(seed=6, options={"desired_goal": np.array([0.2, -0.05, -0.1], np.float32)})
    mirror_state(b, a)
    for attr in ("_filtered_action", "_delayed_action", "_backlash_state"):
        setattr(b, attr, getattr(a, attr)[action_perm()].copy())
    perm, sign = obs_maps(1)
    rng = np.random.default_rng(1)
    early = joints = 0.0
    for t in range(12):
        act = rng.uniform(-0.25, 0.25, 12).astype(np.float32)
        oa, *_ = a.step(act)
        ob, *_ = b.step(act[action_perm()])
        err = np.abs(flatten_obs(oa)[perm] * sign - flatten_obs(ob))
        if t < 5:
            early = max(early, float(err.max()))
        joints = max(joints, float(err[6:30].max()))
    # Yaw then drifts: the model is about 3 mm asymmetric side to side and foot contact is chaotic. Wrong signs would show at once.
    print(f"mirror error: all observations, first 0.1 s {early:.4f}; joints, first 0.24 s {joints:.4f}")
    assert early < 0.02 and joints < 0.03
    a.close()
    b.close()


def fill(env, steps=1600, n_envs=1, priv=0, mirror=None, frames=5):
    rc = RewardConfig.from_env(env)
    dim = 6 + 42 * frames
    rb = GpuHerReplay(4000, n_envs, 256, 2, dim, 12, rc, DEVICE, priv_dim=priv, mirror=mirror)
    rng = np.random.default_rng(0)
    o, _ = env.reset(seed=7)
    pp = env.privileged()
    i, rows = rb.i_rew, []
    for _ in range(steps):
        a = rng.uniform(-0.5, 0.5, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        row = np.zeros((1, rb.width), dtype=np.float32)
        row[0, :dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = flatten_obs(o), flatten_obs(o2), a
        row[0, i:i + 7] = (r, term, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"])
        if priv:
            row[0, rb.i_priv:rb.i_next_priv], row[0, rb.i_next_priv:] = pp, env.privileged()
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([term or trunc]))
        rows.append(row[0].copy())
        o, pp = o2, env.privileged()
        if term or trunc:
            o, _ = env.reset()
            pp = env.privileged()
    return rb, np.array(rows)


def test_replay_mirror_flips_consistently():
    env = make(clean_cfg(history=5))
    env_m = {"obs_perm": obs_maps(5)[0], "obs_sign": obs_maps(5)[1], "act_perm": action_perm()}
    rb, rows = fill(env, mirror=env_m)
    obs, act, rew, nxt, done, _ = [x.cpu().numpy() if torch.is_tensor(x) else x for x in rb.sample()]
    p, s = env_m["obs_perm"], env_m["obs_sign"]
    # Early frames repeat at the start of an episode, so identify a row by all of its frames.
    fingerprint = lambda x: (np.round(x, 5) + 0.0).tobytes()  # + 0.0 turns -0.0 into 0.0
    real = {fingerprint(r[6:rb.obs_dim]): r for r in rows}
    flipped = plain = 0
    for j in range(rb.nb_virtual, rb.batch_size):
        o, a = obs[j], act[j]
        r = real.get(fingerprint(o[6:]))
        if r is not None:
            plain += 1
            assert np.allclose(a, r[rb.i_act:rb.i_rew])
            assert abs(rew[j] - r[rb.i_rew]) < 1e-6
        else:
            back = ((o * s)[np.argsort(p)]).astype(np.float32)
            r = real[fingerprint(back[6:])]
            flipped += 1
            assert np.allclose(a[action_perm()], r[rb.i_act:rb.i_rew])
            assert abs(rew[j] - r[rb.i_rew]) < 1e-6, "mirroring must not change the reward"
            nback = ((nxt[j] * s)[np.argsort(p)]).astype(np.float32)
            assert np.allclose(nback[6:], r[rb.i_next + 6:rb.i_act], atol=1e-5)
    print(f"real samples checked: {plain} plain, {flipped} mirrored")
    assert flipped > 5 and plain > 5
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_everything_together_runs_in_graph_mode():
    env = make(clean_cfg(history=5, obs_noise={"joint_vel": 0.1}))
    op, os_ = obs_maps(5)
    pp, ps = priv_maps()
    rb, _ = fill(env, priv=46, mirror={"obs_perm": op, "obs_sign": os_, "act_perm": action_perm(), "priv_perm": pp, "priv_sign": ps})
    torch.manual_seed(1)
    learner = SacLearner(rb, 216, 12, DEVICE, mode="graph", layer_norm=True, avg_q=True, priv_dim=46, ema_tau=0.01)
    assert learner.critic.qf0[0].in_features == 216 + 46 + 12 and learner.actor.latent_pi[0].in_features == 216
    e0 = {k: v.clone() for k, v in learner.actor_ema.state_dict().items()}
    for _ in range(60):
        learner.update()
    torch.cuda.synchronize()
    assert np.isfinite(list(learner.read_stats().values())).all()
    k = "latent_pi.0.weight"
    ema, online = learner.actor_ema.state_dict()[k], learner.actor.state_dict()[k]
    assert not torch.equal(ema, e0[k]) and not torch.equal(ema, online)
    assert (ema - e0[k]).abs().mean() < (online - e0[k]).abs().mean()
    env.close()
