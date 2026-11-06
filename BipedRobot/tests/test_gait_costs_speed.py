"""The faster slip arithmetic in GaitCostEnv must give exactly the same numbers as np.cross. Run by path from the workspace root."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class CrossEnv(BudgetLocoEnv):
    """substep_loads as it was, with np.cross."""

    def substep_loads(self):
        m, d = self.model, self.data
        fn, weighted = np.zeros(2), np.zeros(2)
        nonfoot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor_geom_id else c.geom1
            mujoco.mj_contactForce(m, d, i, self._f6)
            normal = abs(self._f6[0])
            if other in self.foot_geom_ids:
                k = self.foot_geom_ids.index(other)
                fn[k] += normal
                body = self.foot_body_ids[k]
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, self._v6, 0)
                point = self._v6[3:] + np.cross(self._v6[:3], c.pos - d.xpos[body])
                weighted[k] += normal * float(np.hypot(point[0], point[1]))
            else:
                nonfoot += normal
        return fn, np.where(fn > 1e-9, weighted / np.maximum(fn, 1e-9), 0.0), nonfoot


def make(cls):
    c = yaml.safe_load(open(REPO / "config/v2_deploy_budget.yaml"))
    ec = build_env_config(c, "full", False)
    ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
    env = cls(config=ec)
    env.reward_mode = "track"
    return env


def test_every_number_is_exactly_the_same_over_whole_episodes():
    new, old = make(BudgetLocoEnv), make(CrossEnv)
    rng = np.random.default_rng(0)
    seed = 0
    a, b = new.reset(seed=seed)[0], old.reset(seed=seed)[0]
    loaded = 0
    for _ in range(2500):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        a, ra, ta, tra, ia = new.step(act)
        b, rb, tb, trb, ib = old.step(act)
        assert np.array_equal(flatten_obs(a), flatten_obs(b))
        assert ra == rb and ta == tb and tra == trb
        assert np.array_equal(ia["budget_cost"], ib["budget_cost"]) and ia["cost_slip"] == ib["cost_slip"]
        assert np.array_equal(new.data.qpos, old.data.qpos) and np.array_equal(new.privileged(), old.privileged())
        loaded += ia["cost_slip_n"]
        if ta or tra:
            seed += 1
            new.reset(seed=seed)
            old.reset(seed=seed)
    assert loaded > 1000, "the test must cover loaded feet"
    new.close()
    old.close()
