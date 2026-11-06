"""Check that the training joints and the robot's servos describe the same robot. Runs on the workstation, moves nothing.

Compares ACTUATED_JOINTS in the training env and robot_v2_train.xml against the URDF and servos.yaml in biped_ws:
kinematic chain, joint axes in the world frame, joint offsets, limits, and left/right servo signs. Which way each joint
moves is computed with MuJoCo and must agree with the wording robot_test.py shows on the real robot.

Usage: joint_map.py BIPED_WS OUT_JSON
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from BipedRobot.envs.her_loco_env import ACTUATED_JOINTS  # noqa: E402
from robot_test import DEPLOY_LIMITS, MOTION, SIM  # noqa: E402

XML = ROOT / "BipedRobot/envs/assets/robotV2/robot_v2_train.xml"
CONFIG = ROOT / "BipedRobot/config/v2_budget_survive.yaml"
# training joint suffix -> servos.yaml joint name, for side r or l. In training hip_x/ankle_y are pitch, hip_y/ankle_x roll.
NAMES = {"hip_z": "{}_hip_yaw", "hip_x": "{}_hip_pitch_joint", "hip_y": "{}_hip_roll_joint", "knee": "{}_knee_joint",
         "ankle_y": "{}_ankle_pitch_joint", "ankle_x": "{}_ankle_roll_joint"}
KIND = {"hip_z": "hip_yaw", "hip_x": "hip_pitch", "hip_y": "hip_roll", "knee": "knee", "ankle_y": "ankle_pitch", "ankle_x": "ankle_roll"}


def servo_name(train):
    side, rest = train.split("_", 1)
    return NAMES[rest].format(side)


def read_urdf(path):
    text = re.sub(r"<\?xml.*?\?>", "", Path(path).read_text())
    root = ET.fromstring(re.sub(r"xmlns:\w+=\"[^\"]*\"", "", text).replace("xacro:", ""))
    out = {}
    for j in root.iter("joint"):
        if j.get("type") != "revolute":
            continue
        out[j.get("name")] = {
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "axis": [float(x) for x in j.find("axis").get("xyz").split()],
            "xyz": [float(x) for x in j.find("origin").get("xyz").split()],
            "rpy": [float(x) for x in j.find("origin").get("rpy").split()],
            "lower": float(j.find("limit").get("lower")), "upper": float(j.find("limit").get("upper")),
        }
    return out


def motion_sign(model, data, jid, kind, side, home):
    """Sign of the named foot motion for a small positive joint angle, from forward kinematics at the home pose."""
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
    out_y = -1.0 if side == "r" else 1.0
    points = {"center": np.zeros(3), "toe": np.array([0.05, 0, 0]), "outer": np.array([0, 0.03 * out_y, 0])}

    def world(q):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        return {k: data.xpos[foot] + data.xmat[foot].reshape(3, 3) @ p for k, p in points.items()}

    q = home.copy()
    a = world(q)
    q[model.jnt_qposadr[jid]] += 0.05
    b = world(q)
    d = {k: b[k] - a[k] for k in points}
    if kind == "hip_yaw":
        return "toes_outward" if d["toe"][1] * out_y > 0 else "toes_inward"
    if kind == "hip_pitch":
        return "leg_backward" if d["center"][0] < 0 else "leg_forward"
    if kind == "hip_roll":
        return "leg_outward" if d["center"][1] * out_y > 0 else "leg_inward"
    if kind == "knee":
        return "shin_forward" if d["center"][0] > 0 else "shin_backward"
    if kind == "ankle_pitch":
        return "toes_down" if d["toe"][2] < 0 else "toes_up"
    if kind == "ankle_roll":
        return "outer_edge_up" if d["outer"][2] > 0 else "outer_edge_down"
    raise KeyError(kind)


def main():
    ws, out = Path(sys.argv[1]), Path(sys.argv[2])
    urdf = read_urdf(ws / "robotv2_description/urdf/robot.urdf.xacro")
    sv = yaml.safe_load((ws / "biped_hardware/config/servos.yaml").read_text())
    hw = yaml.safe_load((ws / "biped_hardware/config/hardware.yaml").read_text())
    cfg = yaml.safe_load(CONFIG.read_text())
    default = np.array(cfg["robot"]["default_qpos"])
    frac = float(cfg["action"]["range_fraction"])
    margin = float(hw["safety"]["limit_margin_rad"])

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    home = model.qpos0.copy()
    for i, name in enumerate(ACTUATED_JOINTS):
        home[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = default[i]

    problems, rows = [], []
    order = sv["joint_order"]
    for i, train in enumerate(ACTUATED_JOINTS):
        side, rest = train.split("_", 1)
        name = servo_name(train)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, train)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, train)
        u, s = urdf[name], sv["servos"][name]

        # Axis in the world frame at zero pose, so any rotated body frame in the XML would show up as a difference.
        data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, data)
        axis_xml = data.xaxis[jid].copy()
        if np.abs(axis_xml - np.array(u["axis"])).max() > 1e-6:
            problems.append(f"{train}: axis {axis_xml.tolist()} vs URDF {name} {u['axis']}")
        if any(abs(x) > 1e-9 for x in u["rpy"]):
            problems.append(f"{name}: URDF origin is rotated, so axes are not directly comparable")

        body = model.jnt_bodyid[jid]
        offset_mm = float(np.abs(model.body_pos[body] - np.array(u["xyz"])).max() * 1000)
        if offset_mm > 1.0:
            problems.append(f"{train}: joint offset differs from URDF by {offset_mm:.2f} mm")
        if int(model.actuator_trnid[aid, 0]) != jid:
            problems.append(f"actuator {train} does not drive joint {train}")

        lo, hi = model.jnt_range[jid]
        span_hi, span_lo = hi - default[i], default[i] - lo
        tgt_lo, tgt_hi = default[i] - frac * span_lo, default[i] + frac * span_hi
        if tgt_lo < s["lower"] + margin or tgt_hi > s["upper"] - margin:
            problems.append(f"{train}: policy target {tgt_lo:.3f}..{tgt_hi:.3f} outside servo soft limits {s['lower']}..{s['upper']}")
        if lo < u["lower"] - 1e-5 or hi > u["upper"] + 1e-5:
            problems.append(f"{train}: training range {lo:.3f}..{hi:.3f} outside URDF {u['lower']}..{u['upper']}")

        kind = KIND[rest]
        if np.abs(np.array(DEPLOY_LIMITS[kind]) - np.array([lo, hi])).max() > 1e-5:
            problems.append(f"{train}: robot_test.py limits {DEPLOY_LIMITS[kind]} vs training range {lo:.6f}..{hi:.6f}")
        moves = motion_sign(model, data, jid, kind, side, home)
        if MOTION[kind] != moves:
            problems.append(f"{train}: robot_test.py says + is {MOTION[kind]}, MuJoCo says {moves}")
        rows.append({
            "train_index": i, "train_name": train, "servo_name": name, "servo_index": order.index(name), "servo_id": s["id"],
            "servo_direction": s["direction"], "axis": u["axis"], "default_rad": float(default[i]),
            "train_range": [float(lo), float(hi)], "policy_target_range": [float(tgt_lo), float(tgt_hi)],
            "servo_soft_limits": [s["lower"], s["upper"]], "offset_diff_mm": offset_mm, "positive_motion": moves,
        })

    # Chain order: each joint's parent joint must match between the XML and the URDF.
    link_to_joint = {u["child"]: n for n, u in urdf.items()}
    for r in rows:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r["train_name"])
        parent_body = model.body_parentid[model.jnt_bodyid[jid]]
        pj = model.body_jntadr[parent_body]
        xml_parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, pj) if model.jnt_type[pj] == mujoco.mjtJoint.mjJNT_HINGE else None
        urdf_parent = link_to_joint.get(urdf[r["servo_name"]]["parent"])
        want = servo_name(xml_parent) if xml_parent else None
        if want != urdf_parent:
            problems.append(f"{r['train_name']}: parent joint {xml_parent} vs URDF {urdf_parent}")

    # A left servo is a right servo mounted as its mirror image, so its shaft is the mirrored vector (x, -y, z).
    by = {r["train_name"]: r for r in rows}
    for r in rows:
        if not r["train_name"].startswith("r_"):
            continue
        l = by["l_" + r["train_name"][2:]]
        shaft_r = r["servo_direction"] * np.array(r["axis"])
        shaft_l = l["servo_direction"] * np.array(l["axis"])
        if np.abs(shaft_l - shaft_r * np.array([1, -1, 1])).max() > 1e-6:
            problems.append(f"{r['train_name']}/{l['train_name']}: servo signs are not mirror images")

    # The numbers robot_test.py prints as "what the simulator assumes" must be what training really uses.
    act, val = cfg["actuator"], cfg["validity"]
    assumed = {"delay_range_ms": [1000 * x for x in act["delay_range"]], "servo_lag_tau_ms": 1000 * act["servo_lag_tau"],
               "noise": val["obs_noise"], "bias": val["obs_bias"]}
    for key, value in assumed.items():
        if value != SIM[key]:
            problems.append(f"robot_test.py SIM[{key}] = {SIM[key]}, config says {value}")

    swapped = [(r["train_name"], r["servo_name"]) for r in rows if r["train_index"] != r["servo_index"]]
    print(f"{'#':>2} {'training':10} {'servo #':>7} {'servo name':22} {'ID':>3} {'dir':>4} {'+ moves':16} {'policy target rad':>18} {'servo limits':>16}")
    for r in rows:
        t, s = r["policy_target_range"], r["servo_soft_limits"]
        print(f"{r['train_index']:>2} {r['train_name']:10} {r['servo_index']:>7} {r['servo_name']:22} {r['servo_id']:>3} {r['servo_direction']:>+4} "
              f"{r['positive_motion']:16} {t[0]:>8.3f} ..{t[1]:>7.3f} {s[0]:>7.3f} ..{s[1]:>6.3f}")
    print("\nindices that differ between training and servos.yaml:", swapped)
    print("max joint offset difference, mm:", round(max(r["offset_diff_mm"] for r in rows), 3))
    print("problems:", len(problems))
    for p in problems:
        print(" -", p)
    out.write_text(json.dumps({"rows": rows, "swapped_indices": swapped, "problems": problems,
                               "train_to_servo_index": [r["servo_index"] for r in rows]}, indent=1))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
