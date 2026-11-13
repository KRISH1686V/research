"""Robotis OP3 as the second robot (envs/op3, 12 Nov 2026): model, shapes, costs, buckets and relabeling against brute force, mirror maps, and the
opt-in dispatch in fast/collector.py and fast/train.py. CPU only: run with CUDA_VISIBLE_DEVICES="" from the workspace root."""
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_budget_her as H  # noqa: E402
from test_budget_soft import episodes, soft_want  # noqa: E402
from test_budget_window import brute_window  # noqa: E402
from BipedRobot.envs import her_loco_env as hle  # noqa: E402
from BipedRobot.envs.op3 import make_train_xml, mirror as op3_mirror  # noqa: E402
from BipedRobot.envs.op3.op3_env import ACTION_SCALE, DEFAULT_QPOS, LEG_JOINTS, XML, Op3BudgetEnv, Op3Env  # noqa: E402
from BipedRobot.fast import mirror as biped_mirror  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv, decode_rate  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import env_class, flatten_obs  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "config/op3_budget_r17wS.yaml"
OFF = REPO / "config/op3_budget_off.yaml"
DEVICE = torch.device("cpu")


def op3_config(path=CFG, realism="easy", **budget):
    c = yaml.safe_load(open(path))
    ec = build_env_config(c, realism, False)
    ec["validity"] = copy.deepcopy(c["validity"])
    ec["token_budget"] = {**c["token_budget"], **budget}
    return c, ec


def make(realism="easy", clean=False, **budget):
    """clean: no sensor noise or bias and history 1, for comparing observations exactly."""
    _, ec = op3_config(realism=realism, **budget)
    if clean:
        ec["validity"] = {k: v for k, v in ec["validity"].items() if k not in ("obs_noise", "obs_bias", "encoder_rad", "history")}
        ec["shaping"]["reset_velocity"] = False
    env = Op3BudgetEnv(config=ec)
    env.reward_mode = "track"
    return env


# ---- model -------------------------------------------------------------------------------------------------------------------------------------------
def test_train_xml_is_what_the_generator_writes_and_menagerie_is_kept():
    for path, text in make_train_xml.outputs().items():
        assert path.read_text() == text, f"{path.name} is stale: rerun envs/op3/make_train_xml.py"
    assets = REPO / "envs/assets/robotis_op3"
    assert "Apache License" in (assets / "LICENSE").read_text()[:200]
    m = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
    assert m.nu == 20 and m.njnt == 21, "the Menagerie model itself stays unmodified"


def test_model_loads_with_12_leg_actuators_and_welded_arms():
    env = make()
    m = env.model
    assert env.xml_path == XML and m.nu == 12 and m.njnt == 13 and m.nq == 19
    assert abs(float(m.body_mass.sum()) - 3.14747) < 1e-4, "welding keeps every body's mass"
    assert [m.joint(j).name for j in env.joint_ids] == list(LEG_JOINTS) and [m.actuator(a).name for a in env.actuator_ids] == list(LEG_JOINTS)
    assert np.allclose(m.actuator_gainprm[:, 0], 21.1) and np.allclose(m.actuator_biasprm[:, 1], -21.1)
    assert np.allclose(m.actuator_forcerange, [[-5.0, 5.0]] * 12), "Menagerie's torque limit"
    assert not {"head_pan", "l_sho_roll", "r_el"} & {m.joint(j).name for j in range(m.njnt)}
    assert hle.ACTUATED_JOINTS[0] == "r_hip_z" and hle.FOOT_GEOMS == ["r_foot", "l_foot"] and hle.TORSO_BODY == "torso", "biped names restored"
    assert [len(s) for s in env.foot_geom_sets] == [2, 2]
    env.close()


def test_standing_reset_has_feet_down_level_and_no_self_contact():
    env = make()
    o, _ = env.reset(seed=3, options={"desired_goal": np.array([0, 0, 0, 1, 1, 1], np.float32)})
    assert o["observation"].shape == (42 * 5 + 3,) and o["desired_goal"].shape == (6,) and o["achieved_goal"].shape == (3,)
    assert env.action_space.shape == (12,) and env.privileged().shape == (46,)
    assert abs(env.sole_bottoms().min()) < 1e-9 and env.sole_bottoms().max() < 0.004
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        assert env.floor_geom_id in (c.geom1, c.geom2) and (c.geom1 in env._geom_foot or c.geom2 in env._geom_foot), "only soles touch, nothing touches itself"
    for _ in range(50):
        o, r, term, trunc, info = env.step(np.zeros(12, np.float32))
    assert not term and np.all(env._feet_contact() == 1.0) and 0.24 < env.data.qpos[2] < 0.27
    assert info["r_track"] > 0.9 and abs(r - (info["r_track"] + 0.2 + info["r_smooth"])) < 1e-5, "standing at command 0: tracking, alive 0.2, no progress"
    a, b = (env.data.geom_xpos[env.model.geom(n).id] for n in ("r_sole_long", "l_sole_long"))
    assert 0.09 < abs(a[1] - b[1]) < 0.10, "feet 96 mm apart"
    env.close()


def test_action_map_is_symmetric_in_policy_coordinates():
    env = make()
    assert np.allclose(env._action_to_target(np.zeros(12)), DEFAULT_QPOS)
    rng = np.random.default_rng(0)
    for _ in range(20):
        leg = rng.uniform(-1, 1, 6).astype(np.float32)
        t = env._action_to_target(np.r_[leg, leg])
        assert np.allclose(t[:6], -t[6:], atol=1e-6), "the same action on both legs is a mirror-symmetric pose"
    full = env._action_to_target(np.ones(12, np.float32))
    assert np.allclose(np.abs(full - np.array(DEFAULT_QPOS)), np.tile(ACTION_SCALE, 2), atol=1e-6)
    assert env._action_to_target(-np.ones(12, np.float32))[9] == pytest.approx(0.0, abs=1e-6), "left knee reaches straight, never backwards"
    env.close()


# ---- costs -------------------------------------------------------------------------------------------------------------------------------------------
def independent_loads(env, d):
    """Foot load from all sole boxes and contact-point speed from the body Jacobian (mj_jac at the contact), not the env's v + w x r."""
    m, f6, jacp = env.model, np.zeros(6), np.zeros((3, env.model.nv))
    fn, weighted, other = np.zeros(2), np.zeros(2), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if env.floor_geom_id not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == env.floor_geom_id else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        body = int(m.geom_bodyid[g])
        if body in env.foot_body_ids:
            k = env.foot_body_ids.index(body)
            mujoco.mj_jac(m, d, jacp, None, c.pos, body)
            v = jacp @ d.qvel
            fn[k] += abs(f6[0])
            weighted[k] += abs(f6[0]) * np.hypot(v[0], v[1])
        else:
            other += abs(f6[0])
    return fn, np.where(fn > 1e-9, weighted / np.maximum(fn, 1e-9), 0.0), other


def test_substep_loads_match_an_independent_computation():
    """At states along a random rollout, run the forward pass a physics substep starts with (on a copy) and compare the env's loads with
    independent_loads. The biped's lever arm (slip_reference biped_code) does not give the contact point's speed; the OP3 default does."""
    env = make(clean=True)
    env.reset(seed=1, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1], np.float32)})
    rng = np.random.default_rng(2)
    checked = flying = 0
    biped_err = 0.0
    for _ in range(200):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        if term or trunc:
            env.reset()
            continue
        d = copy.copy(env.data)
        mujoco.mj_forward(env.model, d)
        live, env.data = env.data, d
        try:
            fn, slip, nonfoot = env.substep_loads()
            env.slip_reference = "biped_code"
            _, slip_biped, _ = env.substep_loads()
            env.slip_reference = "contact"
        finally:
            env.data = live
        want_fn, want_slip, want_other = independent_loads(env, d)
        assert np.allclose(fn, want_fn) and abs(nonfoot - want_other) < 1e-9
        assert np.allclose(slip, want_slip, atol=1e-9), (slip, want_slip)
        biped_err = max(biped_err, float(np.abs(slip_biped - want_slip).max()))
        checked += int((fn > 0.5).any())
        flying += int(not (fn > 0.5).any() and nonfoot <= 0.5)
    print(f"largest gap between the biped lever arm and the contact point's speed: {biped_err:.4f} m/s")
    assert checked > 50 and flying > 0, "the test must cover loaded and flight states"
    assert biped_err > 0.01, "the biped's lever arm differs from the contact point's speed (documented in envs/op3/op3_env.py)"
    env.close()


def test_flight_slip_and_jshake_are_the_biped_definitions():
    env = make(clean=True, settle_steps=0)
    env.reset(seed=4, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1], np.float32)})
    subs = []
    env.substep_observer = lambda fn, slip, nonfoot: subs.append((fn.copy(), slip.copy(), nonfoot))
    q = [env.data.qpos[env.joint_qpos_adrs].astype(np.float64).copy()] * 2
    rng = np.random.default_rng(5)
    seen = np.zeros(3)
    for _ in range(120):
        subs.clear()
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        q.append(env.data.qpos[env.joint_qpos_adrs].astype(np.float64).copy())
        loaded = [(fn > 0.5, slip) for fn, slip, _ in subs]
        flight = np.mean([not ld.any() and other <= 0.5 for (ld, _), (_, _, other) in zip(loaded, subs)])
        n = sum(int(ld.sum()) for ld, _ in loaded)
        slip = sum(float(s[ld].sum()) for ld, s in loaded) / n if n else 0.0
        jshake = float(np.mean((q[-1] - 2 * q[-2] + q[-3]) ** 2))
        assert len(subs) == 10, "10 physics substeps of 2 ms per 20 ms control step"
        assert np.allclose(info["budget_cost"], np.float32([flight, min(slip, env.slip_clip), jshake]), rtol=1e-5, atol=1e-12)
        seen += np.asarray(info["budget_cost"]) > 0
        if term or trunc:
            break
    assert (seen > 0).all(), "every cost must be exercised"
    env.close()


def test_settle_steps_zero_the_first_five_costs():
    env = make()
    env.reset(seed=2, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1], np.float32)})
    rng = np.random.default_rng(3)
    costs = [env.step(rng.uniform(-1, 1, 12).astype(np.float32))[4]["budget_cost"].copy() for _ in range(12)]
    assert env.settle_steps == 5 and np.all(np.array(costs[:5]) == 0.0) and np.array(costs[5:]).sum() > 0.0
    env.close()


# ---- bucket and relabeling with the r17wS settings (soft gate, window 100, u_low -0.3, settle 5, alive bonus) ----------------------------------------
def fill(env, data, budget, batch=512):
    rb = BudgetHerReplay(6000, 1, batch, 2, H.env_obs_dim(env), 12, RewardConfig.from_env(env), DEVICE, budget, env.max_episode_steps, relabel_keep=(1, 2))
    i = rb.i_rew
    for d in data:
        info = d["info"]
        row = np.zeros((1, rb.width), dtype=np.float32)
        row[0, :rb.obs_dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = d["obs"], d["nxt"], d["a"]
        row[0, i:] = (d["r"], d["term"], info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        rb.add(torch.as_tensor(row), np.array([d["term"] or d["trunc"]]))
    return rb


def test_env_levels_and_soft_reward_match_brute_force():
    env = make()
    data = H.rollout(env, 1500, amp=0.6)
    over = scaled = 0
    for ep in episodes(data):
        goal = ep[0]["obs"][3:9].astype(np.float64)
        rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
        cap = rate * env.burst_steps
        deficits = brute_window([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate, env.window_steps)
        for t, e in enumerate(ep):
            assert np.allclose(e["obs"][-3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=1e-5)
            assert np.allclose(e["nxt"][-3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=1e-5)
            is_over = bool((deficits[t + 1] > cap).any())
            assert is_over == e["info"]["budget_overdrawn"]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            assert abs(soft_want(base, e["info"]["fell"], deficits[t + 1], cap, env.alive_bonus) - e["r"]) < 1e-5
            over += int(is_over)
            scaled += int(is_over and not e["info"]["fell"] and base > 0)
    assert over > 20 and scaled > 5, "the test must cover overdrawn steps that are scaled"
    env.close()


def test_relabeling_on_op3_transitions_matches_brute_force_exactly():
    c, _ = op3_config()
    budget = c["token_budget"]
    env = make()
    data = H.rollout(env, 2000, amp=0.6)
    rb = fill(env, data, budget)
    index = H.relabeled_truth(env, rb, data)
    checked = over = below_zero = 0
    for _ in range(3):
        obs, _, rew, nxt, done, _ = [x.numpy() for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            goal = obs[j, 3:9].astype(np.float64)
            assert goal[1] == 0.0 and goal[2] == 0.0, "relabel_dims [vx]: vy and wz keep the stored command"
            assert any(f["nxt"][0] == np.float32(goal[0]) for f in ep[t:]), "vx comes from the episode's future"
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = brute_window([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate, env.window_steps)
            assert np.allclose(obs[j, -3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=2e-5)
            assert np.allclose(nxt[j, -3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=2e-5)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            assert abs(soft_want(base, e["info"]["fell"], deficits[t + 1], cap, env.alive_bonus) - rew[j]) < 2e-5
            checked += 1
            over += int((deficits[t + 1] > cap).any())
            below_zero += int((goal[3:] < 0).any())
    assert checked > 1000 and over > 10 and below_zero > 50, "the test must cover overdrawn relabels and u below 0"
    env.close()


# ---- mirror ------------------------------------------------------------------------------------------------------------------------------------------
def test_mirror_maps_are_involutions_and_flip_every_joint():
    for frames, budget in ((1, 0), (5, 3)):
        p, s = op3_mirror.obs_maps(frames, budget)
        bp, bs = biped_mirror.obs_maps(frames, budget)
        assert np.array_equal(p, bp) and len(p) == 6 + budget + 42 * frames + budget
        assert np.array_equal(p[p], np.arange(len(p))) and np.array_equal(s * s[p], np.ones(len(p)))
        start = 6 + budget
        for k in range(frames):
            f = slice(start + 42 * k, start + 42 * k + 42)
            assert np.all(s[f][:24] == -bs[f][:24]) and np.all(s[f][24:] == bs[f][24:])
    pp, ps = op3_mirror.priv_maps(4, 0)
    assert len(pp) == 46 and np.array_equal(pp[pp], np.arange(46)) and np.array_equal(ps * ps[pp], np.ones(46))
    a = op3_mirror.action_perm()
    assert np.array_equal(a[a], np.arange(12)) and np.array_equal(a, np.r_[6:12, 0:6])
    with pytest.raises(ValueError):
        op3_mirror.obs_maps(5, 3, clock=True)


def mirror_state(env, source):
    """Set env to the mirror image of source: every OP3 leg joint changes sign as the legs swap."""
    a = biped_mirror.action_perm()
    q, v = source.data.qpos.copy(), source.data.qvel.copy()
    q[1], v[1] = -q[1], -v[1]
    w, x, y, z = q[3:7]
    q[3:7] = (w, -x, y, -z)
    v[3], v[5] = -v[3], -v[5]
    for i in range(12):
        q[env.joint_qpos_adrs[i]] = -source.data.qpos[source.joint_qpos_adrs[a[i]]]
        v[env.joint_dof_adrs[i]] = -source.data.qvel[source.joint_dof_adrs[a[i]]]
    env.data.qpos[:], env.data.qvel[:] = q, v
    mujoco.mj_forward(env.model, env.data)
    for attr in ("_filtered_action", "_delayed_action", "_backlash_state"):  # joint coordinates
        setattr(env, attr, -getattr(source, attr)[a].copy())
    for attr in ("_action_filter_state", "_prev_action", "_prev_prev_action"):  # policy coordinates
        setattr(env, attr, getattr(source, attr)[a].copy())


def test_mirror_matches_the_simulator():
    a, b = make(clean=True), make(clean=True)
    a.reset(seed=6, options={"desired_goal": np.array([0.2, 0.05, 0.1, 0.5, 0.5, 0.5], np.float32)})
    b.reset(seed=6, options={"desired_goal": np.array([0.2, -0.05, -0.1, 0.5, 0.5, 0.5], np.float32)})
    mirror_state(b, a)
    perm, sign = op3_mirror.obs_maps(1, 3)
    rng = np.random.default_rng(1)
    early = joints = 0.0
    for t in range(12):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        oa, *_ = a.step(act)
        ob, *_ = b.step(act[op3_mirror.action_perm()])
        err = np.abs(flatten_obs(oa)[perm] * sign - flatten_obs(ob))
        if t < 5:
            early = max(early, float(err.max()))
        joints = max(joints, float(err[9:33].max()))
    # The model is almost symmetric (the welded head tilt link sits 19 mm off the centre line); a wrong sign would show at once, at order 0.1-1.
    print(f"OP3 mirror error: all observations, first 0.1 s {early:.4f}; joints, first 0.24 s {joints:.4f}")
    assert early < 0.02 and joints < 0.03
    a.close()
    b.close()


# ---- configs and dispatch ------------------------------------------------------------------------------------------------------------------------------
def test_dispatch_is_opt_in():
    import BipedRobot.fast.train as train
    _, ec = op3_config()
    assert env_class(ec) is Op3BudgetEnv and env_class({**ec, "token_budget": None}) is Op3Env
    assert train._mirror_maps(ec)[1] is op3_mirror.obs_maps and train._mirror_maps({})[1] is biped_mirror.obs_maps
    assert env_class({"token_budget": {"rate_min": [1]}}) is BudgetLocoEnv, "biped configs (no robot_type) keep the biped classes"
    assert train.ROBOT_HASHED["op3"] and all((REPO / f).exists() for f in train.ROBOT_HASHED["op3"])
    with pytest.raises(ValueError):
        env_class({**ec, "robot_type": "go2"})


@pytest.mark.parametrize("bad", [{"validity": {"gait_ref": {"table": "x"}}}, {"shaping": {"foot_flat": {"l1": 1.0}}}, {"shaping": {"ref_imitate": {"weight": 1.0}}},
                                 {"token_budget": {"costs": ["flight", "partsole", "jshake"]}}, {"validity": {"clock_hz": 1.0}}])
def test_biped_only_options_are_refused(bad):
    _, ec = op3_config()
    for k, v in bad.items():
        ec[k] = {**(ec.get(k) or {}), **v}
    with pytest.raises(ValueError):
        Op3BudgetEnv(config=ec)


def test_both_configs_load_at_full_realism_and_differ_only_in_token_budget():
    on, off = yaml.safe_load(open(CFG)), yaml.safe_load(open(OFF))
    assert {k: v for k, v in on.items() if k != "token_budget"} == {k: v for k, v in off.items() if k != "token_budget"}
    b = on["token_budget"]
    assert b["costs"] == ["flight", "slip", "jshake"] and b["overdraft"] == "soft" and b["u_low"] == -0.3 and b["settle_steps"] == 5
    assert b["window_steps"] == 100 and b["burst_steps"] == 100 and "capacity_floor" not in b and "capacity_floor" not in off["token_budget"]
    assert np.all(np.array(off["token_budget"]["rate_min"]) >= 10.0)
    for path in (CFG, OFF):
        _, ec = op3_config(path, realism="full")
        env = env_class(ec)(config=ec)
        env.reward_mode = "track"
        env.set_difficulty(1.0)
        o, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(60):
            o, r, term, trunc, info = env.step(rng.uniform(-0.5, 0.5, 12).astype(np.float32))
            assert np.isfinite(r) and np.all(np.isfinite(o["observation"]))
            if path == OFF:
                assert not info["budget_overdrawn"] and np.allclose(o["observation"][-3:], 1.0, atol=1e-3)
            if term or trunc:
                o, _ = env.reset()
        env.close()
