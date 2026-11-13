"""run_policy.PolicyCore with a slow-yaw policy (validity.slow_yaw_s): the input after the gait clock equals the training env's, step by step."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "BipedRobot/diagnostics/audit_2026-10-31/audit/v2/hardware_tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HW))
RESOLVED = ROOT / "BipedRobot/runs/fast_deploylong7mstraightfit_20261102_seed3/seed_3/resolved_config.yaml"


def env():
    from BipedRobot.evaluation_suite_v2 import load_env_config
    from BipedRobot.fast.collector import env_class
    cfg = load_env_config(RESOLVED)
    return env_class(cfg)(config=cfg, render_mode=None)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if not RESOLVED.exists():
        pytest.skip("straight-step run folder not present")
    import export_policy
    d = tmp_path_factory.mktemp("slowyaw")
    obs_dim, h = 3 + 6 + 42 * 5 + 2 + 1 + 3, 64
    keep = torch.arange(3, obs_dim - 3)
    g = torch.Generator().manual_seed(0)
    actor = {"keep": keep, "latent_pi.0.weight": 0.1 * torch.randn(h, len(keep), generator=g), "latent_pi.0.bias": torch.zeros(h),
             "latent_pi.2.weight": 0.1 * torch.randn(h, h, generator=g), "latent_pi.2.bias": torch.zeros(h),
             "mu.weight": 0.1 * torch.randn(12, h, generator=g), "mu.bias": torch.zeros(12)}
    torch.save({"actor": actor, "obs_dim": obs_dim, "steps": 0}, d / "fake.pt")
    export_policy.main([str(d / "fake.pt"), str(RESOLVED), str(d / "p.npz")])
    return d / "p.npz"


def test_export_carries_the_window(exported):
    z = np.load(exported)
    e = env()
    assert float(z["slow_yaw_s"]) == e.slow_yaw_s == 1.0
    assert int(z["slow_yaw_n"]) == e._slow_yaw.maxlen == 50
    assert float(z["clock_hz"]) == e.clock_hz == 1.0


def test_slow_yaw_input_matches_training_env(exported):
    """Drive the env with random actions, give the runner each step's sensed gyro, and compare the slow-yaw input, including the window filling up and rolling."""
    from run_policy import PolicyCore
    core, e = PolicyCore(str(exported)), env()
    obs, _ = e.reset(seed=3)
    n = e.num_joints
    gscale = e.obs_scale["base_ang_vel"]
    rng = np.random.default_rng(0)
    zeros = np.zeros(12, np.float32)
    for k in range(120):
        o = obs["observation"]
        newest = o[42 * (e.history - 1):42 * e.history]  # frames are oldest first; the newest frame is the last of the five
        gyro = newest[2 * n:2 * n + 3] / gscale
        core.step(core.default.astype(np.float32), zeros, gyro, np.array([0, 0, -1], np.float32), np.array([0.1, 0, 0], np.float32), np.ones(3, np.float32))
        i = 9 + 42 * e.history + 2
        assert core.last_obs[i] == pytest.approx(float(o[42 * e.history + 2]), abs=1e-5), f"step {k}"
        obs, _, term, trunc, _ = e.step(rng.uniform(-0.3, 0.3, 12).astype(np.float32))
        if term or trunc:
            break
    assert k > 60, "the window must have rolled past 50 steps"


def test_policies_without_it_unchanged():
    from run_policy import PolicyCore
    core = PolicyCore(str(HW / "policy_7m_s4.npz"))
    assert core.slow_yaw_n == 0
    core.step(core.default.astype(np.float32), np.zeros(12, np.float32), np.array([0, 0, 0.5], np.float32), np.array([0, 0, -1], np.float32),
              np.array([0.2, 0, 0], np.float32), np.ones(3, np.float32))
    assert len(core.slow_yaw) == 0 and core.last_obs.shape == (core.obs_dim,)
