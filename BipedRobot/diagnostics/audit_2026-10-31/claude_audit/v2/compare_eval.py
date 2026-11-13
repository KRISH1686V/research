"""One evaluation for every method in the comparison (12 Nov 2026): budget-HER (swept u), budgets off, Lagrangian, fixed penalty, episode budget, PPO.
Command 0.1 m/s, medium realism, no pushes, 20 walks of 10 s on test seeds 9800-9819, deterministic policy. Per setting: share upright, forward speed,
sideways drift, torso tilt p95, per-step budget costs exactly as the env counts them (flight, slip, jshake), command and joint shake (5-25 Hz), and the
mean cost of every 100-step (2 s) window, stride 10, so any limit can be checked afterwards: the bucket bounds every window, a Lagrangian only the mean.
Optional robot-like line: in-loop gear play 1.75 deg and 40 ms delay. Optional --switch A>B: the budget changes from A to B at step 250 of each walk
(the knob turned while walking); costs and speed are reported before (steps 100-250) and after (300-500). Writes compare/<arm>_s<seed>.json.
Usage: compare_eval.py ARM SEED [--u 1,1,1 0.5,0.5,0.5 ...] [--robot] [--robot_u ...] [--switch 1,1,1>0,0,0 ...] [--walks 20]"""
import argparse, copy, json, os, sys
import numpy as np, torch
V = os.path.dirname(os.path.abspath(__file__)); R = "/home/rpuser1/yasho/rl/BipedRobot"
sys.path.insert(0, "/home/rpuser1/yasho/rl"); sys.path.insert(0, V)
from BipedRobot.evaluation_suite_v2 import load_env_config
from budget_eval import LoggedEnv
from survive_eval import load_policy
torch.set_num_threads(1)
p = argparse.ArgumentParser()
p.add_argument("arm"); p.add_argument("seed", type=int); p.add_argument("--u", nargs="*", default=["1,1,1"]); p.add_argument("--robot", action="store_true")
p.add_argument("--robot_u", nargs="*", default=None); p.add_argument("--switch", nargs="*", default=[]); p.add_argument("--walks", type=int, default=20)
p.add_argument("--command", type=float, default=0.1, help="forward command m/s (12 Nov: the speed-matched control evaluates budgets-off policies at lower commands)"); p.add_argument("--tag", default="", help="suffix for the output file")
p.add_argument("--seed0", type=int, default=9800, help="first test seed (12 Nov: the primary evaluation uses 10000-10099)"); p.add_argument("--outdir", default="compare", help="output folder under v2/")
a = p.parse_args()
N = f"fast_{a.arm}_20261102_seed{a.seed}"; D = f"{R}/checkpoints/{N}/seed_{a.seed}"
ck = f"{D}/ppo_final.zip" if a.arm.startswith("ppo") else f"{D}/sb3_her_final.pt"
policy, steps = load_policy(ck)
W, STRIDE = 100, 10


def band(x, lo=5, hi=25):
    X = np.fft.rfft(x - x.mean(), axis=0); f = np.fft.rfftfreq(len(x), 0.02); m = (f >= lo) & (f <= hi)
    return np.sqrt((np.abs(X[m]) ** 2).sum(axis=0) * 2) / len(x)


def make_env(robot):
    cfg = copy.deepcopy(load_env_config(f"{R}/runs/{N}/seed_{a.seed}/resolved_config.yaml")); cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = "medium", False, 500
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    if robot: cfg.setdefault("validity", {})["servo_play"] = {"deg": [1.75, 1.75]}
    env = LoggedEnv(config=cfg, render_mode=None); env.reward_mode, env.push_enabled = "track", False
    if robot: env._play_on = True; env.enable_backlash = False; env.enable_delay, env.delay_range, env.nominal_delay = True, (0.04, 0.04), 0.04
    return env


def walk(env, u, s, u_after=None):
    goal = np.array([a.command, 0, 0, *u], np.float32)
    env.log = []; o, _ = env.reset(seed=s, options={"desired_goal": goal}); x0, y0 = float(env.data.qpos[0]), float(env.data.qpos[1])
    C, OD, T, Q, tilt, X = [], [], [], [], [], []; fell = False
    for k in range(500):
        if u_after is not None and k == 250:
            env.set_budget(np.asarray(u_after, np.float32))  # the policy sees the new u in its goal from the next observation
            o["desired_goal"] = np.concatenate((o["desired_goal"][:3], np.asarray(u_after, np.float32))).astype(np.float32)
        o, _, term, trunc, info = env.step(np.asarray(policy(o), np.float32))
        w, x, y, z = env.data.qpos[3:7]; tilt.append(float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))))
        C.append(info["budget_cost"]); OD.append(bool(info["budget_overdrawn"])); X.append(float(env.data.qpos[0]))
        if k >= 100: T.append(env.data.ctrl[env.actuator_ids].copy()); Q.append(env.data.qpos[env.joint_qpos_adrs].copy())
        if term: fell = True; break
        if trunc: break
    return dict(fell=fell, C=np.array(C, np.float64), OD=OD, T=T, Q=Q, tilt=tilt, X=X, x0=x0, y_end=abs(float(env.data.qpos[1]) - y0))


def windows(C):
    """Mean cost of each 100-step window from step 100 on, stride 10: list per cost."""
    C = C[100:]
    if len(C) < W: return [[] for _ in range(3)]
    cs = np.vstack((np.zeros((1, C.shape[1])), np.cumsum(C, axis=0)))
    idx = np.arange(0, len(C) - W + 1, STRIDE)
    return [((cs[idx + W, j] - cs[idx, j]) / W).tolist() for j in range(C.shape[1])]


out = dict(arm=a.arm, seed=a.seed, checkpoint=ck, steps=steps, command=a.command, walks=a.walks, test_seeds=[a.seed0, a.seed0 + a.walks - 1], window_steps=W, window_stride=STRIDE, results=[], switches=[])
plan = [("evaluation", False, a.u)] + ([("robot-like", True, a.robot_u or a.u)] if a.robot else [])
for name, robot, ulist in plan:
    env = make_env(robot)
    for us in ulist:
        u = [float(x) for x in us.split(",")]
        assert len(u) == env.n_cost, (u, env.cost_names)
        Cs, ODs, falls, fwd, side, tilt, CS, JS, WM = [], [], 0, [], [], [], [], [], [[], [], []]
        C0s = []
        for s in range(a.seed0, a.seed0 + a.walks):
            r = walk(env, u, s)
            if len(r["C"]): C0s.append(r["C"])
            falls += r["fell"]; tilt += r["tilt"]
            if len(r["C"]) > 100: Cs.append(r["C"][100:]); ODs += r["OD"][100:]
            if not r["fell"]: fwd.append((r["X"][-1] - r["x0"]) / 10); side.append(r["y_end"])
            if len(r["T"]) > 100: CS.append(float(np.degrees(band(np.array(r["T"]))).mean())); JS.append(float(np.degrees(band(np.array(r["Q"]))).mean()))
            for j, wl in enumerate(windows(r["C"])): WM[j] += wl
        C = np.concatenate(Cs) if Cs else np.full((1, 3), np.nan)
        res = dict(condition=name, u=u, cost_names=list(env.cost_names), up=1 - falls / a.walks, speed=float(np.mean(fwd)) if fwd else 0.0, speed_per_walk=fwd,
                   drift_m=float(np.mean(side)) if side else None, tilt_p95_deg=float(np.percentile(tilt, 95)), cost_per_step=[float(v) for v in np.nanmean(C, axis=0)],
                   overdrawn=float(np.mean(ODs)) if ODs else None, cost_per_step_from0=[float(v) for v in np.nanmean(np.concatenate(C0s), axis=0)] if C0s else None, cmd_shake_deg=float(np.mean(CS)) if CS else None, joint_shake_deg=float(np.mean(JS)) if JS else None,
                   window_mean_cost=WM, window_quantiles={q: [float(np.percentile(w, q)) if w else None for w in WM] for q in (50, 90, 99, 100)})
        out["results"].append(res)
        print(f"{a.arm} s{a.seed} {name:10s} u={u}: up {res['up']:.2f} speed {res['speed']:.3f} | costs {np.round(res['cost_per_step'], 6).tolist()} | worst 2 s window {np.round(res['window_quantiles'][100], 6).tolist()} | shake cmd {res['cmd_shake_deg']} joint {res['joint_shake_deg']}", flush=True)
    env.close()
if a.switch:
    env = make_env(False)
    for sw in a.switch:
        u0, u1 = ([float(x) for x in part.split(",")] for part in sw.split(">"))
        falls, before, after, sp0, sp1 = 0, [], [], [], []
        for s in range(a.seed0, a.seed0 + a.walks):
            r = walk(env, u0, s, u_after=u1)
            falls += r["fell"]
            if len(r["C"]) >= 500:
                before.append(r["C"][100:250].mean(axis=0)); after.append(r["C"][300:500].mean(axis=0))
                sp0.append((r["X"][250] - r["X"][100]) / 3.0); sp1.append((r["X"][499] - r["X"][300]) / 4.0)
        res = dict(switch=sw, u_before=u0, u_after=u1, up=1 - falls / a.walks, cost_before=np.mean(before, axis=0).tolist() if before else None, cost_after=np.mean(after, axis=0).tolist() if after else None,
                   speed_before=float(np.mean(sp0)) if sp0 else None, speed_after=float(np.mean(sp1)) if sp1 else None)
        out["switches"].append(res)
        print(f"{a.arm} s{a.seed} switch {sw}: up {res['up']:.2f} | cost before {np.round(res['cost_before'] or [], 6).tolist()} after {np.round(res['cost_after'] or [], 6).tolist()} | speed {res['speed_before']} -> {res['speed_after']}", flush=True)
    env.close()
os.makedirs(f"{V}/{a.outdir}", exist_ok=True)
json.dump(out, open(f"{V}/{a.outdir}/{a.arm}_s{a.seed}{a.tag}.json", "w"), indent=1)
