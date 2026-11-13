"""run_policy.PolicyCore with a GAIT REFERENCE policy: the exported pattern gives the same joint targets as the training env (fast/valid_env.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "BipedRobot/diagnostics/audit_2026-10-31/audit/v2/hardware_tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HW))
RESOLVED = ROOT / "BipedRobot/runs/fast_deploylong7mheeltoe_20261102_seed3/seed_3/resolved_config.yaml"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if not RESOLVED.exists():
        pytest.skip("heel-toe run folder not present")
    import export_policy
    d = tmp_path_factory.mktemp("ref")
    obs_dim, h = 224, 256
    keep = torch.arange(3, obs_dim - 3)
    g = torch.Generator().manual_seed(0)
    actor = {"keep": keep, "latent_pi.0.weight": 0.1 * torch.randn(h, len(keep), generator=g), "latent_pi.0.bias": torch.zeros(h),
             "latent_pi.2.weight": 0.1 * torch.randn(h, h, generator=g), "latent_pi.2.bias": torch.zeros(h),
             "mu.weight": 0.1 * torch.randn(12, h, generator=g), "mu.bias": torch.zeros(12)}
    torch.save({"actor": actor, "obs_dim": obs_dim, "steps": 0}, d / "fake.pt")
    export_policy.main([str(d / "fake.pt"), str(RESOLVED), str(d / "p.npz")])
    return d / "p.npz"


def env():
    from BipedRobot.evaluation_suite_v2 import load_env_config
    from BipedRobot.fast.collector import env_class
    cfg = load_env_config(RESOLVED)
    e = env_class(cfg)(config=cfg, render_mode=None)
    e.reset(seed=0)
    return e


def test_export_carries_the_pattern(exported):
    z = np.load(exported)
    e = env()
    assert np.array_equal(z["ref_table"], e._ref_table)
    assert float(z["ref_residual"]) == e.ref_residual == 0.25 and float(z["ref_ramp_s"]) == e.ref_ramp_s == 1.0
    assert float(z["clock_hz"]) == e.clock_hz == 1.0


def test_targets_match_training_env(exported):
    """Same filtered action, same issue time: the robot's joint targets equal the env's, through the ramp and the cycle wrap."""
    from run_policy import PolicyCore
    core, e = PolicyCore(str(exported)), env()
    rng = np.random.default_rng(1)
    for k in [0, 1, 7, 25, 49, 50, 51, 77, 130, 499]:
        filtered = rng.uniform(-1, 1, 12).astype(np.float32)
        core.ref_k, core.ref_scale = k, 1.0
        e._ref_t = k * e.dt
        np.testing.assert_allclose(core.to_target(filtered), e._action_to_target(filtered), atol=1e-6, err_msg=f"k={k}")


def test_step_reads_pattern_at_issue_time_and_scale_shrinks_it(exported):
    from run_policy import PolicyCore
    core = PolicyCore(str(exported))
    zeros = np.zeros(3, np.float32)
    q = core.default.astype(np.float32)
    for k in range(60):
        _, target = core.step(q, np.zeros(12, np.float32), zeros, np.array([0, 0, -1], np.float32), np.array([0.2, 0, 0], np.float32), np.ones(3, np.float32), scale=0.0)
        np.testing.assert_allclose(target, np.clip(core.default, np.maximum(core.lo, core.ctrl_min), np.minimum(core.hi, core.ctrl_max)), atol=1e-6)
    assert core.ref_k == 60 and core.clock_k == 60
    half = PolicyCore(str(exported))
    half.ref_scale = 0.3
    full = PolicyCore(str(exported))
    t = 0.37 + 2.0
    np.testing.assert_allclose(half.ref_pose(t) - half.default, 0.3 * (full.ref_pose(t) - full.default), atol=1e-12)


def test_plain_policies_unchanged():
    from run_policy import PolicyCore
    p = HW / "policy_7m_s4.npz"
    core = PolicyCore(str(p))
    assert core.ref_table is None
    f = np.linspace(-1, 1, 12).astype(np.float32)
    expect = np.clip(np.where(f >= 0, core.default + f * core.fraction * (core.hi - core.default), core.default + f * core.fraction * (core.default - core.lo)), core.ctrl_min, core.ctrl_max)
    np.testing.assert_allclose(core.to_target(f), expect, atol=0)
