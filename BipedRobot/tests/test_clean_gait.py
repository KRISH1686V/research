"""Options added 6 Nov 2026 for the clean-gait arms (fast/valid_env.py, fast/budget_replay.py, fast/mirror.py): gait_always, swing_contact, straight,
duty_range and her.relabel_dims. Each is off unless named. CPU only, so it can run beside training on the GPU."""
import copy
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import env_class, flatten_obs  # noqa: E402
from BipedRobot.fast.mirror import priv_maps  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "config/v2_deploy_budget_alpha30_straightflat_fit.yaml"
CPU = torch.device("cpu")


def make(shaping=None, validity=None, realism="medium"):
    c = yaml.safe_load(BASE.read_text())
    c["shaping"].update(shaping or {})
    c["validity"].update(validity or {})
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        ec[k] = copy.deepcopy(c[k])
    ec["push_enabled"] = False
    return env_class(ec)(config=ec, render_mode=None), c


def test_defaults_unchanged():
    env, _ = make()
    env.reset(seed=0)
    assert env.priv_signed == 0 and env.privileged().shape == (46,) and env._duty == 1.0
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert "swing_contact" not in info and "straight_match" not in info
    assert info["r_gait_still"] != info["r_gait_move"] or info["r_gait_move"] == 0.0
    p0, s0 = priv_maps()
    p1, s1 = priv_maps(4, 0)
    assert np.array_equal(p0, p1) and np.array_equal(s0, s1)


def test_gait_always_copies_the_moving_terms():
    env, _ = make({"gait_always": True})
    env.reset(seed=1)
    for _ in range(30):
        _, _, _, _, info = env.step(np.random.default_rng(0).uniform(-0.3, 0.3, 12).astype(np.float32))
        assert info["r_gait_still"] == info["r_gait_move"]


def test_swing_contact_counts_a_loaded_swing_foot_and_skips_the_margins():
    """Standing still with both feet loaded: each step, the foot the clock swings is loaded for every substep, unless the phase is inside the margin."""
    m = 0.05
    env, _ = make({"swing_contact": {"weight": 1.0, "load_frac": 0.05, "margin": m}})
    env.reset(seed=2)
    checked = {0.0: 0, 1.0: 0}
    for k in range(60):
        phase = env._phase  # the phase the policy saw; the step's substeps use it
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
        fn = env.substep_loads()[0]
        if not (fn > 0.05 * env._weight_n).all():
            continue  # settling: a foot not yet carrying weight
        inside = (m <= phase < 0.5 - m) or (0.5 + m <= phase < 1.0 - m)
        assert info["swing_contact"] == (1.0 if inside else 0.0), (k, phase, info["swing_contact"])
        checked[info["swing_contact"]] += 1
    assert checked[1.0] >= 30 and checked[0.0] >= 2, checked
    assert env._weight_n == float(env.model.body_mass.sum()) * 9.81


def test_straight_term_and_critic_inputs():
    env, _ = make({"straight": {"weight": 1.0, "yaw_std": 0.10, "y_std": 0.05}})
    env.reset(seed=3)
    assert env.straight_error() == (0.0, 0.0)
    assert env.priv_signed == 2 and env.privileged().shape == (48,)
    yaw0, x0, y0 = env._start_pose
    d = env.data
    # turn the robot 0.1 rad and move it 0.05 m to its left of the start heading
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([math.cos(0.05), 0, 0, math.sin(0.05)]), d.qpos[3:7].copy())
    d.qpos[3:7] = q
    d.qpos[0] += -math.sin(yaw0) * 0.05
    d.qpos[1] += math.cos(yaw0) * 0.05
    mujoco.mj_forward(env.model, d)
    err, side = env.straight_error()
    assert abs(err - 0.1) < 1e-9 and abs(side - 0.05) < 1e-9
    info = env._get_info(action=np.zeros(12, np.float32))
    assert abs(info["straight_match"] - math.exp(-1.0)) < 1e-9
    env._get_obs_dict()
    assert np.allclose(env.privileged()[-2:], [0.1, 0.05], atol=1e-6)
    _, sign = priv_maps(4, 2)
    assert list(sign[-6:]) == [1, 1, 1, 1, -1, -1]


def test_duty_scales_torque_and_speed_together():
    env, _ = make(validity={"duty_range": [0.5, 0.5]})
    env.reset(seed=4)
    assert env._duty == 0.5
    assert env._stall == 0.5 * env._stall0 and env._cap == 0.5 * env._cap0 and env._curve_w0 == 0.5 * env._curve_w00
    assert abs(env.privileged()[45] - 0.5 * float(np.mean(env._torque_scale))) < 1e-6
    env.set_difficulty(0.0)
    env.reset(seed=5)
    assert env._duty == 1.0 and env._stall == env._stall0
    wide, _ = make(validity={"duty_range": [0.5, 1.0]})
    seen = set()
    for s in range(20):
        wide.reset(seed=s)
        assert 0.5 <= wide._duty <= 1.0
        seen.add(round(wide._duty, 6))
    assert len(seen) > 10


def test_relabel_keeps_the_stored_turn_and_sideways_command():
    env, c = make()
    dim = sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())
    rb = BudgetHerReplay(4000, 1, 512, 2, dim, 12, RewardConfig.from_env(env), CPU, c["token_budget"], relabel_keep=(1, 2))
    plain = BudgetHerReplay(4000, 1, 512, 2, dim, 12, RewardConfig.from_env(env), CPU, c["token_budget"])
    rng = np.random.default_rng(0)
    goal = np.array([0.08, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
    o, _ = env.reset(seed=6, options={"desired_goal": goal})
    i = rb.i_rew
    for _ in range(1500):
        a = rng.uniform(-0.6, 0.6, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        row = np.zeros((1, rb.width), np.float32)
        row[0, :dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = flatten_obs(o), flatten_obs(o2), a
        row[0, i:i + 10] = (r, term, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        for b in (rb, plain):
            b.add(torch.as_tensor(row), np.array([term or trunc]))
        o = o2
        if term or trunc:
            o, _ = env.reset(options={"desired_goal": goal})
    torch.manual_seed(0)
    obs = rb.sample()[0]
    V = rb.nb_virtual
    assert torch.all(obs[:V, 4:6] == 0.0), "vy and wz keep the stored command"
    assert obs[:V, 3].std() > 0.0, "vx is still relabelled"
    torch.manual_seed(0)
    obs_plain = plain.sample()[0]
    assert obs_plain[:V, 4:6].abs().max() > 0.0, "without the option vy and wz are relabelled"


def test_raw_action_rate_sees_the_policy_output_before_the_filter():
    env, _ = make({"raw_action_rate": 0.05})
    env.set_smoothness_ramp(1.0)
    env.reset(seed=7)
    a = np.full(12, 0.5, np.float32)
    _, _, _, _, info = env.step(a)
    assert info["raw_action_rate"] == 0.0, "nothing to compare with on the first step"
    _, _, _, _, info = env.step(-a)
    assert abs(info["raw_action_rate"] - 12.0) < 1e-9, "sum over 12 joints of (−0.5 − 0.5)^2"
    _, _, _, _, info = env.step(-a)
    assert info["raw_action_rate"] == 0.0
    plain, _ = make()
    plain.reset(seed=7)
    assert "raw_action_rate" not in plain.step(a)[4]


def test_phase_margin_allows_double_support_only_near_the_change_of_feet():
    """Standing on both feet: without the margin each step scores 0 (one foot right); with it, steps within the margin of a change of feet score +w."""
    w, m = 0.5, 0.05
    env, _ = make({"phase_margin": m})
    plain, _ = make()
    for e in (env, plain):
        e.reset(seed=8)
    seen_near = seen_far = 0
    for _ in range(80):
        phase = env._phase
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
        _, _, _, _, info0 = plain.step(np.zeros(12, np.float32))
        if not (env._feet_contact() > 0.5).all() or not (plain._feet_contact() > 0.5).all():
            continue
        near = min(env._phase % 0.5, 0.5 - env._phase % 0.5) < m
        diff = info["r_gait_move"] - info0["r_gait_move"]
        if near:
            seen_near += 1
            assert abs(diff - w) < 1e-6, (phase, diff)
        else:
            seen_far += 1
            assert abs(diff) < 1e-6, (phase, diff)
    assert seen_near >= 4 and seen_far >= 30, (seen_near, seen_far)


def test_free_ends_lets_heel_and_toe_tilt_only_near_the_change_of_feet():
    fe = 0.12
    env, _ = make({"foot_flat": {"l1": 1.0, "l2": 2.0, "axes": "both", "free_ends": fe}})
    env.reset(seed=9)
    env.set_smoothness_ramp(1.0)
    rng = np.random.default_rng(1)
    near = far = 0
    for _ in range(60):
        _, _, _, _, info = env.step(rng.uniform(-0.4, 0.4, 12).astype(np.float32))
        down = env._feet_contact() > 0.5
        full = np.sin(env.sole_angle()) * down
        roll = np.abs(env.sole_slope()[:, 1]) * down
        expect = roll if min(env._phase % 0.5, 0.5 - env._phase % 0.5) < fe else full
        cost = float(np.sum(np.abs(expect)) + 2.0 * np.sum(expect * expect))
        assert abs(info["foot_flat_cost"] - cost) < 1e-9
        if min(env._phase % 0.5, 0.5 - env._phase % 0.5) < fe:
            near += 1
        else:
            far += 1
    assert near >= 10 and far >= 20


def test_upright_rewards_a_level_torso_and_falls_off_with_lean():
    env, _ = make({"upright": {"weight": 0.5, "pitch_deg": 4, "roll_deg": 8}})
    env.reset(seed=10)
    d = env.data
    base = d.qpos[3:7].copy()
    for pitch_deg, roll_deg in ((0, 0), (4, 0), (10, 0), (0, 8), (-4, 0)):
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.radians([roll_deg, pitch_deg, 0.0]), "xyz")
        d.qpos[3:7] = q
        mujoco.mj_forward(env.model, d)
        p, r = env.torso_pitch_roll()
        assert abs(math.degrees(p) - pitch_deg) < 1e-6 and abs(math.degrees(r) - roll_deg) < 1e-6
        info = env._get_info(action=np.zeros(12, np.float32))
        expect = math.exp(-(pitch_deg / 4) ** 2 - (roll_deg / 8) ** 2)
        assert abs(info["upright_match"] - expect) < 1e-6, (pitch_deg, roll_deg, info["upright_match"])
    d.qpos[3:7] = base
    plain, _ = make()
    plain.reset(seed=10)
    assert "upright_match" not in plain.step(np.zeros(12, np.float32))[4]


def test_com_offset_moves_the_torso_mass_within_range():
    env, _ = make(validity={"com_offset_m": [0.012, 0.006]})
    base = env._torso_ipos0.copy()
    seen = []
    for s in range(20):
        env.reset(seed=s)
        dx, dy, dz = env.model.body_ipos[env.torso_body_id] - base
        assert abs(dx) <= 0.012 + 1e-12 and abs(dy) <= 0.006 + 1e-12 and dz == 0.0
        seen.append(dx)
    assert np.std(seen) > 0.003
    env.reset(seed=3); a = env.model.body_ipos[env.torso_body_id].copy()
    env.reset(seed=3); assert np.array_equal(a, env.model.body_ipos[env.torso_body_id]), "same seed, same offset"
    env.set_difficulty(0.0); env.reset(seed=4)
    assert np.array_equal(env.model.body_ipos[env.torso_body_id], base)
    plain, _ = make()
    plain.reset(seed=5)
    assert np.array_equal(plain.model.body_ipos[plain.torso_body_id], plain._torso_ipos0)


def test_com_support_scores_the_mass_over_the_loaded_feet():
    env, _ = make({"com_support": {"weight": 0.3, "std_m": 0.015}})
    env.reset(seed=11)
    for _ in range(25):
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
    d = env.com_ahead_of_support()
    assert d is not None and abs(d) < 0.03, d  # standing: mass roughly over the feet
    assert abs(info["com_support_match"] - math.exp(-(d / 0.015) ** 2)) < 1e-9


def test_push_range_draws_a_force_for_each_push():
    """Over 6 push intervals: every push force lies in [2, 9] N and they differ; without the option the force stays the configured 3 N."""
    def pushes(env):
        env.push_enabled = True
        env.reset(seed=12)
        out = set()
        for _ in range(6 * env.dr_push_interval + 3):
            env.step(np.zeros(12, np.float32))
            f = round(float(np.linalg.norm(env._push_force)), 6)
            if f > 0:
                out.add(f)
        return sorted(out)
    env, _ = make(validity={"push_range_n": [2.0, 9.0]})
    got = pushes(env)
    assert len(got) >= 4 and all(2.0 - 1e-6 <= f <= 9.0 + 1e-6 for f in got), got
    plain, _ = make()
    assert pushes(plain) == [round(float(plain.dr_push_magnitude), 6)]


def test_clock_stance_half_is_the_old_clock_and_058_gives_double_support():
    old, _ = make()
    for p in np.linspace(0, 0.999, 200):
        r, l = old.clock_stance_feet(p)
        assert r == (p < 0.5) and l == (p >= 0.5)
    env, _ = make(validity={"clock_stance": 0.58})
    both = [p for p in np.linspace(0, 0.999, 1000) if all(env.clock_stance_feet(p))]
    assert abs(len(both) / 1000 - 0.16) < 0.01, "2 x 8% of the cycle with both feet down"
    assert all(min(abs(p - c) % 1.0 for c in (0.0, 0.5, 1.0)) <= 0.04 + 1e-9 for p in both), "centred on the changes of feet"
    none = [p for p in np.linspace(0, 0.999, 1000) if not any(env.clock_stance_feet(p))]
    assert none == []
    for p in np.linspace(0, 0.999, 300):  # the swing bump lifts only a foot the clock has in swing, never both, and never above the height
        env._phase = p
        t = env.clock_swing_target()
        r, l = env.clock_stance_feet(p)
        assert (t[0] == 0.0) == r and (t[1] == 0.0) == l and t.max() <= env.clock_height_m + 1e-12
    old._phase = 0.25
    assert np.allclose(old.clock_swing_target(), [0.0, old.clock_height_m])


def test_hip_yaw_hold_and_speed_margin():
    env, _ = make({"hip_yaw_hold": {"weight": 0.3, "std_deg": 4}, "speed_margin": {"weight": 0.5, "frac": 0.8, "noload_rad_s": 5.3}})
    assert [env.model.joint(env.joint_ids[k]).name for k in env._hip_z_idx] == ["r_hip_z", "l_hip_z"]
    env.reset(seed=13)
    d = env.data
    qa, da = np.asarray(env.joint_qpos_adrs), np.asarray(env.joint_dof_adrs)
    d.qpos[qa] = env.default_qpos
    d.qvel[da] = 0.0
    mujoco.mj_forward(env.model, d)
    info = env._get_info(action=np.zeros(12, np.float32))
    assert abs(info["hip_yaw_match"] - 1.0) < 1e-9 and info["speed_over"] == 0.0
    d.qpos[qa[env._hip_z_idx[0]]] += math.radians(4.0)  # one hip turned by one std: mean of (1, 0) -> exp(-0.5)
    d.qvel[da[3]] = 0.8 * 5.3 * 1.5  # the right knee at 1.5 x the margin speed: (1.5 - 1)^2
    mujoco.mj_forward(env.model, d)
    info = env._get_info(action=np.zeros(12, np.float32))
    assert abs(info["hip_yaw_match"] - math.exp(-0.5)) < 1e-6 and abs(info["speed_over"] - 0.25) < 1e-9


def test_stance_width_rewards_the_target_spacing_and_is_off_by_default():
    env, _ = make({"stance_width": {"weight": 0.5, "width_m": 0.08, "std_m": 0.04}})
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    a, b = (env.data.geom_xpos[g] for g in env.foot_geom_ids)
    assert abs(info["stance_width_m"] - abs(a[1] - b[1])) < 0.005, "with the heading near zero it is the y distance"
    assert 0.08 < info["stance_width_m"] < 0.12, "standing is about 104 mm"
    assert abs(info["stance_width_match"] - math.exp(-((info["stance_width_m"] - 0.08) / 0.04) ** 2)) < 1e-9
    env.stance_width_m = info["stance_width_m"]  # a turned robot reads the same width
    w0 = env.stance_width_now()
    env.data.qpos[3:7] = [math.cos(0.25), 0, 0, math.sin(0.25)]
    mujoco.mj_forward(env.model, env.data)
    assert abs(env.stance_width_now() - w0) < 0.01
    plain, _ = make()
    plain.reset(seed=0)
    assert "stance_width_match" not in plain.step(np.zeros(12, np.float32))[4]


def test_clock_bend_ignores_a_rigid_tilt_and_sees_a_bent_leg():
    env, _ = make({"clock_bend": {"weight": 1.2, "height_m": 0.022, "std_m": 0.02}, "clock_height": {"weight": 0.3, "height_m": 0.025, "std_m": 0.012}})
    env.reset(seed=0)
    m, d = env.model, env.data
    for adr, q in zip(env.joint_qpos_adrs, env.default_qpos):
        d.qpos[adr] = q
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    assert abs(env.foot_bend()) < 1e-6, "zero in the standing pose"
    d.qpos[3:7] = [math.cos(math.radians(10)), math.sin(math.radians(10)), 0, 0]  # the whole robot rolled 20 deg
    mujoco.mj_forward(m, d)
    assert abs(env.foot_bend()) < 1e-6, "a rigid tilt earns nothing"
    assert abs(env.foot_clearance()[0] - env.foot_clearance()[1]) > 0.02, "while the floor measure sees the same tilt as lift"
    d.qpos[3:7] = [1, 0, 0, 0]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[a]) for a in env.joint_dof_adrs]
    d.qpos[env.joint_qpos_adrs[names.index("r_knee")]] -= 0.4
    mujoco.mj_forward(m, d)
    assert env.foot_bend() > 0.005, "bending the right knee raises the right foot along the torso"
    d.qpos[env.joint_qpos_adrs[names.index("r_knee")]] += 0.4
    d.qpos[env.joint_qpos_adrs[names.index("l_knee")]] -= 0.4
    mujoco.mj_forward(m, d)
    assert env.foot_bend() < -0.005, "and the left the other way"


def test_clock_bend_target_follows_the_clock_and_is_off_by_default():
    env, _ = make({"clock_bend": {"weight": 1.2, "height_m": 0.022, "std_m": 0.02}, "clock_height": {"weight": 0.3, "height_m": 0.025, "std_m": 0.012}})
    env.reset(seed=0)
    env._phase = 0.25  # first half: left foot swings
    assert abs(env.clock_bend_target() + 0.022) < 1e-9
    env._phase = 0.75
    assert abs(env.clock_bend_target() - 0.022) < 1e-9
    assert env.clock_height_m == 0.025, "the floor-height target is left as it was"
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert abs(info["clock_bend_match"] - math.exp(-((info["foot_bend_m"] - env.clock_bend_target()) / 0.02) ** 2)) < 0.6
    assert 0.0 <= info["clock_bend_match"] <= 1.0
    plain, _ = make()
    plain.reset(seed=0)
    assert "clock_bend_match" not in plain.step(np.zeros(12, np.float32))[4]


R7 = {"bend_ramp": {"weight": 1.2, "height_m": 0.02}, "knee_lift": {"weight": 0.5, "rad": 0.26}, "min_clearance": {"penalty": 0.5, "height_m": 0.008}}


def standing(env):
    env.reset(seed=0)
    for adr, q in zip(env.joint_qpos_adrs, env.default_qpos):
        env.data.qpos[adr] = q
    env.data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)


def test_midswing_terms_pay_only_at_midswing_and_read_the_right_leg():
    env, _ = make(R7)
    standing(env)
    names = [env.model.joint(j).name for j in env.joint_ids]
    d = env.data
    d.qpos[env.joint_qpos_adrs[names.index("r_knee")]] -= 0.4  # bend the right knee 23 deg
    mujoco.mj_forward(env.model, d)
    assert env.foot_bend() > 0.005
    flex = -d.qpos[[env.joint_qpos_adrs[k] for k in env._knee_idx]]
    assert flex[0] - flex[1] > 0.3, "the right knee is the first of _knee_idx and flexion is positive"
    # rigid tipping earns zero on the lift measure
    env2, _ = make(R7)
    standing(env2)
    env2.data.qpos[3:7] = [math.cos(math.radians(10)), math.sin(math.radians(10)), 0, 0]
    mujoco.mj_forward(env2.model, env2.data)
    assert abs(env2.foot_bend()) < 1e-6
    # the terms arrive only inside 40-60% of the swing: step and read which phases pay
    env3, _ = make(R7)
    env3.reset(seed=1)
    paid = {}
    for _ in range(100):
        _, _, term, trunc, info = env3.step(np.zeros(12, np.float32))
        f = (env3._phase % 0.5) / 0.5
        paid[round(f, 3)] = info["r_midswing"]
        if term or trunc:
            break
    assert all(v == 0.0 for f, v in paid.items() if f < 0.4 or f > 0.6), paid
    assert any(v != 0.0 for f, v in paid.items() if 0.4 <= f <= 0.6), paid


def test_midswing_terms_are_off_by_default():
    plain, _ = make()
    plain.reset(seed=0)
    info = plain.step(np.zeros(12, np.float32))[4]
    for k in ("r_midswing", "ref_match", "bend_rel_m", "knee_extra_rad"):
        assert k not in info
    assert plain._imit_table is None and plain.rsi_prob == 0.0


def table_file(tmp_path, n=50, period=1.0):
    """A reference with a known shape: the left knee bent more in the first half of the cycle (left swings), the right in the second, as fast/gait_ref.py gives."""
    env, _ = make()
    env.reset(seed=0)
    names = [env.model.joint(j).name for j in env.joint_ids]
    t = np.tile(env.default_qpos, (n, 1)).astype(np.float64)
    ph = np.arange(n) / n
    t[:, names.index("l_knee")] -= 0.4 * np.clip(np.sin(2 * np.pi * ph), 0, None)
    t[:, names.index("r_knee")] -= 0.4 * np.clip(-np.sin(2 * np.pi * ph), 0, None)
    f = tmp_path / "ref.npz"
    np.savez(f, table=t.astype(np.float32), period_s=np.float64(period), speed_m_s=np.float64(0.1))
    return str(f), t, names


def test_reference_start_only_in_training_and_phase_matches_the_clock(tmp_path):
    f, table, names = table_file(tmp_path)
    env, _ = make({"ref_imitate": {"table": f, "weight": 1.5, "std_rad": 0.2, "rsi_prob": 1.0}})
    goal = np.array([0.1, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
    env.reset(seed=3, options={"desired_goal": goal})
    assert env._clock_k == 1, "a reset naming its goal starts at phase 0"
    assert abs(env.data.qpos[env.joint_qpos_adrs[names.index("l_knee")]] - env.default_qpos[names.index("l_knee")]) < 0.03, "and standing"
    seen = 0
    for seed in range(8):
        o, _ = env.reset(seed=seed)
        p = env._phase
        o2 = o["observation"]
        k = 42 * env.history
        assert abs(o2[k] - math.sin(2 * math.pi * p)) < 1e-5 and abs(o2[k + 1] - math.cos(2 * math.pi * p)) < 1e-5, "the clock the policy reads is the reference's phase"
        from BipedRobot.fast.gait_ref import lookup
        q = env.data.qpos[env.joint_qpos_adrs]
        want = lookup(table, p)
        assert np.max(np.abs(q - want)) < 1e-5, "joints on the reference at exactly that phase"
        assert abs(float(np.min(env.foot_clearance()))) < 0.003, "lowest foot on the floor"
        assert np.max(np.abs(env._action_filter_state)) <= 1.0
        seen += p > 0.1
    assert seen >= 4, "phases are random"
    # holding the reference earns the full reward, and the swing foot is the clock's
    env._phase = 0.25
    names_l, names_r = env.joint_qpos_adrs[names.index("l_knee")], env.joint_qpos_adrs[names.index("r_knee")]
    from BipedRobot.fast.gait_ref import lookup
    env.data.qpos[env.joint_qpos_adrs] = lookup(table, 0.25)
    mujoco.mj_forward(env.model, env.data)
    assert env.ref_match() > 0.999999
    assert -env.data.qpos[names_l] > -env.data.qpos[names_r], "at phase 0.25 the clock swings the left foot, and the reference bends the left knee"


def test_new_terms_do_not_depend_on_the_goal_and_relabel_matches_the_stored_reward(tmp_path):
    from BipedRobot.fast.replay import relabel_reward
    f, _, _ = table_file(tmp_path)
    shaping = {**R7, "gait_always": True, "ref_imitate": {"table": f, "weight": 1.5, "std_rad": 0.2, "rsi_prob": 0.0}, "stance_width": {"weight": 0.5, "width_m": 0.07, "std_m": 0.03}}
    outs = []
    for vx in (0.03, 0.12):
        env, _ = make(shaping)
        env.reset(seed=5, options={"desired_goal": np.array([vx, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)})
        rng = np.random.default_rng(0)
        outs.append([env.step(rng.uniform(-0.2, 0.2, 12).astype(np.float32))[4] for _ in range(60)])
    for a, b in zip(*outs):
        for k in ("r_gait_move", "r_midswing", "ref_match"):
            assert abs(a[k] - b[k]) < 1e-6, k
        assert a["r_gait_still"] == a["r_gait_move"], "gait_always, so a relabeled command cannot switch them off"
    rc = RewardConfig.from_env(env)
    g, ach = torch.tensor([[0.09, 0.0, 0.0]]), torch.tensor([[0.07, 0.0, 0.01]])
    i = outs[0][10]
    t = lambda x: torch.tensor([float(x)])
    base = relabel_reward(rc, ach, g, t(i["r_smooth"]), t(i["r_gait_move"]), t(i["r_gait_still"]), t(i["r_flight"]), t(0.0))
    low = relabel_reward(rc, ach, torch.tensor([[0.05, 0.0, 0.0]]), t(i["r_smooth"]), t(i["r_gait_move"]), t(i["r_gait_still"]), t(i["r_flight"]), t(0.0))
    zero = relabel_reward(rc, ach, ach, t(0.0), t(0.0), t(0.0), t(0.0), t(0.0))
    add = float(i["r_smooth"]) + float(i["r_flight"]) + float(i["r_gait_move"])
    assert abs(float(base - relabel_reward(rc, ach, g, t(0.0), t(0.0), t(0.0), t(0.0), t(0.0))) - add) < 1e-5, "the gait terms pass through relabeling unchanged"
    assert abs(float(low - relabel_reward(rc, ach, torch.tensor([[0.05, 0.0, 0.0]]), t(0.0), t(0.0), t(0.0), t(0.0), t(0.0))) - add) < 1e-5, "whichever vx it is relabeled to"
    assert float(zero) > 0.0


GATE = {"comfort_gate": {"height_m": 0.02, "frac": 0.5, "min_bump_m": 0.003}, "upright": {"weight": 1.0, "pitch_deg": 4, "roll_deg": 10}, "stance_width": {"weight": 0.5, "width_m": 0.07, "std_m": 0.03},
        "straight": {"weight": 1.0, "yaw_std": 0.10, "y_std": 0.05}, "hip_yaw_hold": {"weight": 0.3, "std_deg": 4}}


def test_comfort_gate_closes_for_standing_and_tipping_and_opens_for_a_bent_leg():
    env, _ = make(GATE)
    standing(env)
    names = [env.model.joint(j).name for j in env.joint_ids]
    env._phase = 0.75  # the middle of the right foot's swing
    assert env.comfort_gate_now() < 1e-6, "standing still earns nothing at mid-swing"
    env.data.qpos[3:7] = [math.cos(math.radians(10)), math.sin(math.radians(10)), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    assert env.comfort_gate_now() < 1e-6, "nor does tipping the whole robot 20 deg"
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qpos[env.joint_qpos_adrs[names.index("r_knee")]] -= 0.6
    mujoco.mj_forward(env.model, env.data)
    assert env.foot_bend() > 0.01
    assert env.comfort_gate_now() == 1.0, "a bent swing leg opens it"
    env._phase = 0.25  # now the left foot should swing, and the right is the raised one
    assert env.comfort_gate_now() == 0.0, "the wrong foot raised earns nothing"
    env._phase = 0.5001  # the start of a swing: the bump is small, so it is open
    assert env.comfort_gate_now() == 1.0


def test_comfort_gate_takes_the_comfort_out_of_the_reward_and_is_off_by_default():
    a, _ = make(GATE)
    b, _ = make({k: v for k, v in GATE.items() if k != "comfort_gate"})
    a.reset(seed=4); b.reset(seed=4)
    zero = np.zeros(12, np.float32)
    seen = 0
    for _ in range(40):
        ia, ib = a.step(zero)[4], b.step(zero)[4]
        comfort = sum(w * ib[k] for w, k in ((1.0, "upright_match"), (0.5, "stance_width_match")))
        comfort += b.straight_weight * ib["straight_match"] if b.straight_weight else 0.0
        comfort += b.hip_yaw_weight * ib["hip_yaw_match"] if b.hip_yaw_weight else 0.0
        assert abs((ib["r_smooth"] - ia["r_smooth"]) - (1.0 - ia["comfort_gate"]) * comfort) < 1e-5
        seen += ia["comfort_gate"] < 0.5
    assert seen > 10, "a robot given no actions mostly gets the gate closed"
    assert b.hip_yaw_weight == 0.3 and b.straight_weight == 1.0, "all four comfort terms are in this test"
    assert "comfort_gate" not in b.step(zero)[4]


def test_stance_band_pays_inside_and_costs_outside_and_is_off_by_default():
    env, _ = make({"stance_width": {"weight": 0.5, "band_m": [0.085, 0.105], "std_m": 0.02, "penalty": 1.0, "penalty_span_m": 0.02}})
    env.reset(seed=0)
    zero = np.zeros(12, np.float32)
    calls = []
    real = env.stance_width_now
    for w, out in ((0.095, 0.0), (0.085, 0.0), (0.105, 0.0), (0.075, 0.010), (0.125, 0.020), (0.150, 0.045)):
        env.stance_width_now = lambda w=w: w
        env.reset(seed=0)
        i = env.step(zero)[4]
        assert abs(i["stance_width_out_m"] - out) < 1e-9, w
        assert abs(i["stance_width_match"] - math.exp(-(out / 0.02) ** 2)) < 1e-9, w
        calls.append((w, i["r_smooth"]))
    inside = calls[0][1]
    assert calls[3][1] < inside - 0.5 * 0.5, "10 mm outside costs half the penalty"
    assert calls[5][1] < calls[3][1], "further outside costs more"
    env.stance_width_now = real
    old, _ = make({"stance_width": {"weight": 0.5, "width_m": 0.08, "std_m": 0.04}})
    old.reset(seed=0)
    assert old.stance_band is None and "stance_width_out_m" not in old.step(zero)[4]


def test_tilt_limit_costs_past_the_line_only():
    env, _ = make({"tilt_limit": {"deg": 8, "span_deg": 8, "penalty": 1.0}})
    plain, _ = make()
    res = {}
    for roll in (0.0, 4.0, 12.0, 16.0, 30.0):
        for e in (env, plain):
            e.reset(seed=0)
            standing(e)
            e.data.qpos[3:7] = [math.cos(math.radians(roll) / 2), math.sin(math.radians(roll) / 2), 0, 0]
            mujoco.mj_forward(e.model, e.data)
        ti = float(np.arccos(np.clip(env.data.xmat[env.torso_body_id][8], -1, 1)))
        res[roll] = np.degrees(ti)
        assert abs(res[roll] - roll) < 1e-6
        want = min(max((roll - 8.0) / 8.0, 0.0), 1.0)
        assert abs(float(np.clip((ti - env.tilt_limit_rad) / env.tilt_limit_span, 0, 1)) - want) < 1e-6
    env.reset(seed=0)
    assert "tilt_rad" in env.step(np.zeros(12, np.float32))[4]
    plain.reset(seed=0)
    assert "tilt_rad" not in plain.step(np.zeros(12, np.float32))[4] and plain.tilt_limit_penalty == 0.0

