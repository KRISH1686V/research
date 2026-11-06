# ACTION PLAN FOR CLAUDE: ACHIEVING PERFECT BIPEDAL WALKING KINEMATICS

**Date:** 5 November 2026  
**Target:** Clean, deliberate, human/robot-like bipedal walking:
1. **Upright, Balanced Center of Mass (CoM):** CoM actively balanced upright ($\theta_{\text{pitch}} \in [-2^\circ, +2^\circ]$) rather than pitch-diving or "falling forward" into hurried steps.
2. **Exact Cadence:** Exactly **2 steps in 1 second** ($2.0\text{ Hz}$ stepping frequency, $T_{\text{step}} = 0.50\text{ s}$, $10\text{ cm}$ stride length at $0.20\text{ m/s}$) — NO 6 Hz shuffling.
3. **Straight-Forward Trajectory:** Locked straight-line vector forward with minimal lateral drift ($|y| \le 0.05\text{ m}$) and yaw lock ($|\Delta \psi| \le 0.05\text{ rad} \approx 2.8^\circ$).
4. **Hardware Safe:** Peak torque $\le 2.94\text{ N}\cdot\text{m}$, safe continuous $\le 1.0\text{ N}\cdot\text{m}$ ($<5\%$ torque saturation to protect ST3215 servos).  
**Hardware Deadline:** Monday, 9 November 2026 (Physical test in `/home/rpuser1/biped_robot_test/`)

---

## The Core Problem Defined

The current 7.2M Budget-HER policy (`policy_7m_s4`) meets all benchmark survival and velocity criteria, but visually it exhibits an unrefined, hurried gait:
- **Forward Pitch Dive ("Falling Forward"):** The torso tilts forward by $8^\circ\text{--}12^\circ$. Instead of actively balancing the Center of Mass upright, the robot lets the CoM fall forward ahead of the Center of Pressure (CoP), hurriedly moving its feet to catch the tumble.
- **Cadence Trapped in 6 Hz Shuffle:** Takes **$\sim 6.0\text{ steps/sec}$** ($T_{\text{step}} \approx 0.16\text{ s}$) with micro-strides of only **$3.5\text{ cm}$** and foot clearance of only **$10\text{--}17\text{ mm}$**.
- **Contact Chatter:** $2\text{--}3.5$ micro-bounces per second.

Conversely, the upstream author's model (`step_30000000.pt`) marches with deliberate high steps, but has a **fatal flaw**: **$95.60\%$ torque saturation**, which will burn out our real ST3215 servos in $<30\text{ seconds}$.

We need to bridge this gap: **achieve an upright, balanced CoM walking at exactly 2 steps per second while keeping torque saturation safe ($<5\%$)**.

---

## The 6 Specific Engineering Bottlenecks & Fixes

### Bottleneck 1: Upright CoM Balance vs. Passive Forward Tumble
* **The Physics:** Under the Linear Inverted Pendulum Model (LIPM):
  $$\ddot{x}_{\text{CoM}} = \frac{g}{z_c}(x_{\text{CoM}} - x_{\text{CoP}})$$
  When the torso pitches forward, $x_{\text{CoM}}$ moves ahead of the foot support polygon ($x_{\text{CoP}}$), creating forward gravitational acceleration. The policy scurries its feet at 6 Hz to "catch" the fall.
* **The Requirement:** The user explicitly demands: *"CoM should be balanced instead of falling or CoM going in forward direction."*
* **Concrete Action for Claude:**
  1. Enforce strict torso uprightness: constraint $|\theta_{\text{pitch}}| \le 0.05\text{ rad}$ ($2.8^\circ$) and pelvis height $z_{\text{pelvis}} \ge 0.20\text{ m}$.
  2. Penalize angular momentum rate $\dot{H}_{\text{CoM}}$ or torso forward pitch bias so the robot balances the CoM directly over the stance foot during weight transfer.

---

### Bottleneck 2: Cadence Control — Exactly 2 Steps in 1 Second ($2.0\text{ Hz}$)
* **The Physics:** 
  - Target cadence: **$2.0\text{ steps/sec}$** ($T_{\text{step}} = 0.50\text{ s}$ per step; full left-right stride cycle = $1.0\text{ s}$).
  - At commanded forward speed $v_x^* = 0.20\text{ m/s}$, the required step length is:
    $$L_{\text{step}} = v_x^* \times T_{\text{step}} = 0.20\text{ m/s} \times 0.50\text{ s} = 0.10\text{ m} = 10.0\text{ cm per step}$$
  - Swing duration: $T_{\text{swing}} \approx 0.40\text{--}0.45\text{ s}$.
  - Double support duration: $T_{\text{double}} \approx 0.05\text{--}0.10\text{ s}$.
* **The Flaw:** Without a cadence governor, the policy takes 6 rapid 3.5 cm steps.
* **Concrete Action for Claude:**
  1. **Phase Clock Conditioning:** Activate the gait clock input $[\sin(2\pi f t), \cos(2\pi f t)]$ with fixed frequency $f = 2.0\text{ Hz}$ (already partially drafted in `v2_deploy_budget_clock.yaml`).
  2. **Swing Duration Budget:** Add a token-bucket budget penalizing contact switching faster than $2.2\text{ Hz}$ or single-support phases shorter than $0.35\text{ s}$.

---

### Bottleneck 3: Lateral Gravity Moment during 0.45s Single Support
* **The Physics:** Holding a single-stance leg for $0.45\text{ s}$ requires supporting the entire $4.5\text{ kg}$ mass on one foot. Shifting the CoM laterally by $6\text{ cm}$ creates:
  $$\tau_{\text{stance\_roll}} \approx 4.5 \times 9.81 \times 0.06 \approx 2.65\text{ N}\cdot\text{m} \quad (90\% \text{ of } 2.94\text{ N}\cdot\text{m} \text{ stall})$$
* **The Flaw:** Static balancing causes ST3215 roll motors to clamp and overheat.
* **Concrete Action for Claude:**
  1. Leverage **dynamic limit-cycle walking**: the lateral CoM sway should dynamically oscillate synchronously with the 2 Hz cadence ($0.5\text{ s}$ period), transferring momentum across legs rather than locking the stance roll servo statically.
  2. Soften stance ankle roll PD stiffness slightly ($K_p = 50\text{--}60$) to allow compliance during roll transitions.

---

### Bottleneck 4: PD Position Gain Saturation ($K_p = 80$, $K_d = 1.0$)
* **The Physics:** $\tau = \text{clip}(K_p(\theta_{\text{des}} - \theta) - K_d \dot{\theta}, -2.94, +2.94)$.
  With $K_p = 80$, a tracking error of just **$2.1^\circ$ ($0.037\text{ rad}$)** completely saturates the actuator to $2.94\text{ N}\cdot\text{m}$.
* **The Flaw:** When the policy demands lifting the leg at 2 Hz, servo tracking lag produces hard torque saturation.
* **Concrete Action for Claude:**
  1. Test reducing pitch joint position gains to $K_p \in [45, 60]$ or apply torque-rate bounding.
  2. Maintain action filter $\alpha = 0.3$ to eliminate abrupt step jumps.

---

### Bottleneck 5: Foot Contact Chatter & Impact Transients
* **The Physics:** Ground reaction forces show $2.1\text{--}3.5\text{ s}^{-1}$ micro-chatter bounces.
* **Concrete Action for Claude:**
  1. Add touchdown velocity compliance or contact impact penalty in the budget formulation.
  2. Ensure feet land flat with ankle pitch compliance before weight transfer.

---

### Bottleneck 6: Policy Exploration vs. Residual Trajectory Prior
* **Concrete Options for Claude:**
  1. **Option A (Refine Existing High-Lift Checkpoint):** Evaluate **`fast_deploywingait_20261102_seed4`**, which already achieves **$61.8\text{ mm}$ foot clearance** with safe **$4.5\%$ torque saturation** ($0.92\text{ N}\cdot\text{m}$ mean torque). Measure its cadence and tune it toward 2 steps/sec.
  2. **Option B (Residual RL / CPG Warm-Start):** Implement a simple $2.0\text{ Hz}$ Bézier foot-lift trajectory generator $\theta_{\text{nominal}}(t)$ ($5\text{ cm}$ foot clearance, $10\text{ cm}$ stride) and learn residual policy $\Delta \theta(s)$:
     $$\theta(t) = \theta_{\text{nominal}}(t) + \pi_{\text{residual}}(s)$$
     This guarantees upright CoM and exactly 2 steps/sec from step 1.

---

### Bottleneck 7: Straight-Line Forward Path (Eliminating Yaw Drift & Lateral Skew)
* **The Physics:** In previous diagnostic runs over 10-second trials, the robot drifts laterally by $0.15\text{--}0.49\text{ m}$ and accumulates yaw rotations of $0.25\text{--}0.52\text{ rad}$ ($14^\circ\text{--}30^\circ$) due to minor left-right foot contact asymmetries.
* **The Requirement:** The user explicitly demands: *"walk in a straight forward pattern, not 6 steps per second."*
* **Concrete Action for Claude:**
  1. **Active Heading Hold:** Incorporate gyro / BNO085 IMU yaw feedback into the policy observation space ($e_{\psi} = \psi_{\text{ref}} - \psi$), enabling active heading regulation during training.
  2. **Lateral Velocity & Yaw Budget:** Enforce token-bucket constraints on lateral drift velocity $|v_y| \le 0.05\text{ m/s}$ and yaw rate $|\omega_z| \le 0.10\text{ rad/s}$ so the policy walks strictly forward along the x-axis without curving or snaking.
  3. **Step Symmetry Constraint:** Penalize differential step length $|L_{\text{left}} - L_{\text{right}}|$ to eliminate the differential ground reaction thrust that causes yaw rotation.

---

## Step-by-Step Execution Plan for Claude

1. **Step 1 (Immediate - 0 GPU hours):** Run an instrumented kinematic analysis on `deploywingait_s4` to see its exact cadence, stride length, and joint trajectory curves compared to `policy_7m_s4`.
2. **Step 2:** Decide between:
   - Exporting `deploywingait_s4` directly to `/home/rpuser1/biped_robot_test/` as the primary demo candidate for Monday's physical test.
   - Or adding a 1.8 Hz phase clock / swing duration constraint to `v2_deploy_budget_alpha30.yaml` for a fresh 5-seed run.
3. **Step 3:** Confirm the physical robot safety limits (never exceed $1.0\text{ N}\cdot\text{m}$ continuous or $2.94\text{ N}\cdot\text{m}$ peak) before Monday's hardware trials.

