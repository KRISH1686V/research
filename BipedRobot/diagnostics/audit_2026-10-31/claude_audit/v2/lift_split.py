"""Where a swing foot's lift comes from. At each swing's highest point, against the 0.2 s before lift-off: the foot's rise; the rise of the hip on that side (body tipping or rising); how much the leg shortened, hip to foot (bending the leg).
Usage: lift_split.py ARM:SEED:COMMAND[:CHECKPOINT] ... [--json OUT]. Heuristic, seeds 9300-9303, medium, mid budget, 10 s. CPU only, no rendering. Written 8 Nov 2026."""
import copy, json, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
import numpy as np, mujoco
V = Path(__file__).resolve().parent; R = V.parents[3]
sys.path.insert(0, str(R.parent)); sys.path.insert(0, str(V))
from BipedRobot.evaluation_suite_v2 import load_env_config
from budget_eval import BUDGETS, LoggedEnv, pad_budget
from survive_eval import load_policy

def run(arm, seed, cmd, ck="sb3_her_final.pt", seeds=(9300, 9301, 9302, 9303), steps=500):
    N = f"fast_{arm}_20261102_seed{seed}"
    cfg = copy.deepcopy(load_env_config(f"{R}/runs/{N}/seed_{seed}/resolved_config.yaml"))
    cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = "medium", False, steps
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    policy, _ = load_policy(f"{R}/checkpoints/{N}/seed_{seed}/{ck}")
    rows = []
    for ts in seeds:
        env = LoggedEnv(config=cfg, render_mode=None); env.reward_mode = "track"; env.push_enabled = False; env.log = []
        obs, _ = env.reset(seed=ts, options={"desired_goal": np.array([cmd, 0, 0, *pad_budget(BUDGETS["all_mid"], env)], np.float32)})
        m, d = env.model, env.data
        hips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("r_hip", "l_lower_hip")]
        jn = [m.joint(j).name for j in env.joint_ids]; ka = [env.joint_qpos_adrs[jn.index(n)] for n in ("r_knee", "l_knee")]
        T = []
        for _ in range(steps):
            obs, _, term, _, _ = env.step(np.asarray(policy(obs), np.float32))
            if term: break
            feet = [d.geom_xpos[g].copy() for g in env.foot_geom_ids]
            T.append((env._feet_contact().copy(), [f[2] for f in feet], [d.xpos[h][2] for h in hips], [np.linalg.norm(d.xpos[h] - f) for h, f in zip(hips, feet)], [-d.qpos[a] for a in ka]))
        env.close()
        if not T: continue
        con = np.array([t[0] for t in T]) > 0.5; fz, hz, ll, fx = (np.array([t[i] for t in T]) for i in (1, 2, 3, 4))
        for leg in (0, 1):
            k = 50; s0 = 50
            while k < len(T) - 1:
                if con[k, leg]: k += 1; continue
                e = k
                while e < len(T) and not con[e, leg]: e += 1
                if e >= len(T): break
                if e - k >= 5:
                    p = k + int(np.argmax(fz[k:e, leg])); b = max(0, k - 10)
                    rows.append((fz[p, leg] - np.mean(fz[b:k, leg]), hz[p, leg] - np.mean(hz[b:k, leg]), np.mean(ll[b:k, leg]) - ll[p, leg], np.degrees(fx[(k + e) // 2, leg] - fx[(s0 + k) // 2, leg]) / 1000.0))
                k = e; s0 = e
    a = 1000 * np.array(rows) if rows else np.zeros((1, 4))
    lift, hip, leg, knee = (float(np.median(a[:, i])) for i in range(4))
    return dict(arm=arm, seed=seed, command=cmd, swings=len(rows), foot_lift_mm=lift, hip_rise_mm=hip, hip_share=hip / lift if lift > 0 else None, leg_shorten_mm=leg, knee_extra_deg=knee)

if __name__ == "__main__":
    args = sys.argv[1:]; out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; del args[i:i + 2]
    res = []
    for spec in args:
        p = spec.split(":"); r = run(p[0], int(p[1]), float(p[2]), *(p[3:4])); res.append(r)
        print(f"{r['arm'].replace('deploylong7m', '')} s{r['seed']} @{r['command']}: {r['swings']} swings | foot lift {r['foot_lift_mm']:5.1f} mm | hip on that side rises {r['hip_rise_mm']:5.1f} mm | leg shortens {r['leg_shorten_mm']:5.1f} mm | knee bends {r['knee_extra_deg']:5.1f} deg more at mid-swing than mid-stance")
    if out: Path(out).write_text(json.dumps(res))
