"""Builds robot_v2_train.xml from robot_v2.xml for HERLocoEnv. Geometry and link masses stay as the V2 design; see the printed summary.

- Names become the ones HERLocoEnv and evaluation_suite_v2 expect. Here r_* is the physical right leg (-y).
- base_link mass is set to the hardware repo's URDF value, so the total is 1.66 kg as in its README.
- Joint ranges use the designer's deploy limits (trained_models/walk_v1/deploy_config.json), tighter than the URDF.
- Actuators: P-only position servo from the published 12 V STS3215 fit (kt 2.2617, R 4.1581) with firmware kp 32 and error gain 0.166:
  kp = 32 * 0.166 * 12 * kt / R = 34.7 Nm/rad, kv = kt^2 / R = 1.23 Nm s/rad. Armature 0.0234, friction 0.149 Nm.
  The fit is from AsterisCrack/BipedRobot's TODO_SIM2REAL.md and has not been validated on this hardware.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
KT, R, KP_FW, ERR_GAIN, VIN = 2.2617, 4.1581, 32.0, 0.166, 12.0
KP = KP_FW * ERR_GAIN * VIN * KT / R
KV = KT ** 2 / R
BODY = {"base_link": "torso", "r_hip_upper_link_1": "r_upper_hip", "r_hip_lower_link_1": "r_hip", "r_thigh_link_1": "r_upper_leg", "r_calf_link_1": "r_lower_leg",
        "r_ankle_link_1": "r_ankle", "r_foot_link_1": "r_foot", "l_hip_upper_link_1": "l_upper_hip", "l_hip_lower_link_1": "l_lower_hip", "l_thigh_link_1": "l_upper_leg",
        "l_calf_link_1": "l_lower_leg", "l_ankle_link_1": "l_ankle", "l_foot_link_1": "l_foot"}
JOINT = {"hip_yaw": "hip_z", "hip_roll_joint": "hip_y", "hip_pitch_joint": "hip_x", "knee_joint": "knee", "ankle_roll_joint": "ankle_x", "ankle_pitch_joint": "ankle_y"}
LIMITS = {"hip_z": (-0.785398, 0.785398), "hip_y": (-0.436332, 0.785398), "hip_x": (-1.570796, 0.523599), "knee": (-2.094395, 0.087266), "ankle_x": (-1.047198, 1.047198), "ankle_y": (-0.610865, 0.872665)}
ORDER = ["hip_z", "hip_x", "hip_y", "knee", "ankle_y", "ankle_x"]

tree = ET.parse(HERE / "robot_v2.xml")
root = tree.getroot()
world = root.find("worldbody")
for body in root.iter("body"):
    if body.get("name") in BODY:
        body.set("name", BODY[body.get("name")])
for joint in root.iter("joint"):
    side, _, rest = (joint.get("name") or "").partition("_")
    if rest in JOINT:
        short = JOINT[rest]
        joint.set("name", f"{side}_{short}")
        joint.set("range", "{} {}".format(*LIMITS[short]))
        joint.attrib.pop("actuatorfrcrange", None)
torso = next(b for b in world.iter("body") if b.get("name") == "torso")
# robot_v2.xml gives the torso no inertial, so MuJoCo derives 0.72 kg from the mesh. Use the hardware repo's URDF values.
for geom in torso.findall("geom"):
    geom.set("mass", "0")
inertial = ET.Element("inertial", {"pos": "0.002427 -0.001742 0.038179", "mass": "0.633299", "fullinertia": "0.001313 0.001431 0.000566 -0.000041 -0.000066 0.00001"})
torso.insert(0, inertial)
free = torso.find("freejoint")
torso.remove(free)
world.remove(torso)
holder = ET.SubElement(world, "body", {"name": "root", "pos": torso.get("pos")})
holder.append(free)
torso.set("pos", "0 0 0")
holder.append(torso)
for tag in ("default", "actuator", "option"):
    for e in root.findall(tag):
        root.remove(e)
ET.SubElement(root, "option", {"timestep": "0.005", "integrator": "implicitfast"})
default = ET.SubElement(root, "default")
ET.SubElement(default, "joint", {"damping": "0", "armature": "0.02342", "frictionloss": "0.14907"})
act = ET.SubElement(root, "actuator")
for side in "rl":
    for short in ORDER:
        n = f"{side}_{short}"
        ET.SubElement(act, "position", {"name": n, "joint": n, "kp": f"{KP:.3f}", "kv": f"{KV:.4f}", "ctrlrange": "-3.141592 3.141592", "forcerange": "-2.94 2.94"})
ET.indent(tree)
tree.write(HERE / "robot_v2_train.xml")
print(f"kp {KP:.2f} kv {KV:.3f}")
