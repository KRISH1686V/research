
"""
HER-Loco Environment: Actuator-realistic Goal-Conditioned Gymnasium Environment
for 12-DOF Bipedal Robot (MuJoCo).

Features:
- Goal-conditioned observation space (proprioception, achieved_goal, desired_goal).
- ST3215 actuator modeling: torque limits (2.94 N*m), action latency buffer (10-30 ms),
  first-order servo lag filter, gear backlash/deadband.
- Domain randomization: link mass, ground friction, action latency, motor strength, horizontal push perturbations.
- Sliding-window body-frame velocity estimation for achieved goal.
- Comprehensive per-step violation logging for adaptive reward correction.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from BipedRobot.envs.rewards.shared_reward import compute_reward_batch

DEFAULT_XML_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets", "robot", "Robot_description", "urdf", "robot_mujoco.xml")
)

# Actuated joint and actuator names in order matching robot_mujoco.xml
ACTUATED_JOINTS: List[str] = [
    "r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x",
    "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x",
]

FOOT_GEOMS: List[str] = ["r_foot", "l_foot"]
FOOT_BODIES: List[str] = ["r_foot", "l_foot"]
TORSO_BODY: str = "torso"
ROOT_BODY: str = "root"
FLOOR_GEOM: str = "floor"


class HERLocoEnv(gym.Env):
    """
    Goal-conditioned 12-DOF biped locomotion environment with actuator dynamics.
    
    Observation space:
        Dict({
            'observation': Box(42,)  # joint pos (12), joint vel (12), base ang vel (3), proj gravity (3), prev action (12)
            'achieved_goal': Box(3,) # [vx, vy, wz] body-frame velocity averaged over sliding window
            'desired_goal': Box(3,)  # [vx*, vy*, wz*] commanded target velocity
        })
    Action space:
        Box(12,) in [-1, 1] mapped to joint position ranges
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        raw_config = dict(config) if config else {}
        default_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/her_loco_config.yaml")
        if os.path.exists(default_yaml):
            try:
                import yaml
                with open(default_yaml, "r") as f:
                    file_cfg = yaml.safe_load(f)
                if isinstance(file_cfg, dict):
                    for k, v in file_cfg.items():
                        if k not in raw_config:
                            raw_config[k] = v
            except Exception:
                pass
        self.config = raw_config
        self.render_mode = render_mode
        self.max_episode_steps = int(self.config.get("max_episode_steps", max_episode_steps))

        # Resolve XML path (default to BipedRobot robot_mujoco.xml)
        if xml_path is None:
            xml_path = self.config.get("xml_path", DEFAULT_XML_PATH)
        self.xml_path = os.path.abspath(xml_path)
        if not os.path.exists(self.xml_path):
            # Fallback search
            if os.path.exists(DEFAULT_XML_PATH):
                self.xml_path = DEFAULT_XML_PATH
            else:
                raise FileNotFoundError(f"MuJoCo XML model not found at {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation timings: 50 Hz control loop (dt = 0.02s)
        self.dt: float = float(self.config.get("control_dt", 0.02))
        self.frame_skip: int = int(self.config.get("frame_skip", 2))
        self.sim_dt: float = self.dt / self.frame_skip
        self.model.opt.timestep = self.sim_dt

        # Actuator indices and joint properties
        self.num_joints: int = len(ACTUATED_JOINTS)
        self.actuator_ids: List[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATED_JOINTS
        ]
        self.joint_ids: List[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ACTUATED_JOINTS
        ]
        self.joint_qpos_adrs: List[int] = [
            int(self.model.jnt_qposadr[jid]) for jid in self.joint_ids
        ]
        self.joint_dof_adrs: List[int] = [
            int(self.model.jnt_dofadr[jid]) for jid in self.joint_ids
        ]

        # Geometry and body IDs
        self.torso_body_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self.root_body_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        self.floor_geom_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        self.foot_geom_ids: List[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOMS
        ]
        self.foot_body_ids: List[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODIES
        ]

        # Joint limits and nominal ranges
        self.jnt_min = np.array([self.model.jnt_range[jid, 0] for jid in self.joint_ids], dtype=np.float32)
        self.jnt_max = np.array([self.model.jnt_range[jid, 1] for jid in self.joint_ids], dtype=np.float32)
        self.ctrl_min = np.array([self.model.actuator_ctrlrange[aid, 0] for aid in self.actuator_ids], dtype=np.float32)
        self.ctrl_max = np.array([self.model.actuator_ctrlrange[aid, 1] for aid in self.actuator_ids], dtype=np.float32)
        
        configured_default = self.config.get("robot", {}).get("default_qpos")
        if configured_default is None:
            configured_default = np.zeros(self.num_joints, dtype=np.float32)
        if len(configured_default) != self.num_joints:
            raise ValueError(f"robot.default_qpos must contain {self.num_joints} values")
        self.default_qpos = np.asarray(configured_default, dtype=np.float32)
        action_cfg = self.config.get("action", {})
        self.action_range_fraction = float(action_cfg.get("range_fraction", 0.5))
        obs_scale_cfg = self.config.get("obs_scale", {})
        self.obs_scale = {
            "joint_pos": float(obs_scale_cfg.get("joint_pos", 1.0)),
            "joint_vel": float(obs_scale_cfg.get("joint_vel", 1.0)),
            "base_ang_vel": float(obs_scale_cfg.get("base_ang_vel", 1.0)),
            "proj_gravity": float(obs_scale_cfg.get("proj_gravity", 1.0)),
            "prev_action": float(obs_scale_cfg.get("prev_action", 1.0)),
        }

        # Feetech ST3215 Actuator specifications
        actuator_cfg = self.config.get("actuator", {})
        self.torque_limit: float = float(actuator_cfg.get("torque_limit", 2.94))  # 2.94 N*m
        self.model.actuator_forcerange[self.actuator_ids, 0] = -self.torque_limit
        self.model.actuator_forcerange[self.actuator_ids, 1] = self.torque_limit
        self.nominal_delay: float = float(actuator_cfg.get("delay_sec", 0.020))   # 20 ms
        self.delay_range: Tuple[float, float] = tuple(actuator_cfg.get("delay_range", (0.010, 0.030)))
        self.servo_lag_tau: float = float(actuator_cfg.get("servo_lag_tau", 0.020)) # 20 ms filter
        self.backlash_deadband: float = float(actuator_cfg.get("backlash_deadband", 0.005)) # rad
        self.enable_backlash: bool = bool(actuator_cfg.get("enable_backlash", True))
        self.enable_delay: bool = bool(actuator_cfg.get("enable_delay", True))
        self.enable_lag: bool = bool(actuator_cfg.get("enable_lag", True))

        # Sliding window for achieved goal computation (default 0.5s = 25 steps at 50Hz)
        self.window_sec: float = float(self.config.get("achieved_goal_window_sec", 0.5))
        self.window_len: int = max(1, int(round(self.window_sec / self.dt)))
        self.velocity_window: deque[np.ndarray] = deque(maxlen=self.window_len)
        budget_cfg = self.config.get("budget_goal", {})
        self.budget_goal_enabled = bool(budget_cfg.get("enabled", False))
        self.budget_cost_names = tuple(
            budget_cfg.get(
                "costs",
                ("torque_rms", "torso_tilt", "foot_slip", "action_rate"),
            )
        )
        self.budget_d_max = np.asarray(
            budget_cfg.get("d_max", [1.0] * len(self.budget_cost_names)),
            dtype=np.float32,
        )
        self.budget_min = np.asarray(
            budget_cfg.get("min_budget", [0.1] * len(self.budget_cost_names)),
            dtype=np.float32,
        )
        if len(self.budget_d_max) != len(self.budget_cost_names):
            raise ValueError("budget_goal.d_max must match budget_goal.costs")
        if len(self.budget_min) != len(self.budget_cost_names):
            raise ValueError("budget_goal.min_budget must match budget_goal.costs")
        self.cost_windows = {
            name: deque(maxlen=self.window_len) for name in self.budget_cost_names
        }

        # Domain randomization parameters
        dr_cfg = self.config.get("domain_randomization", {})
        self.dr_enabled: bool = bool(dr_cfg.get("enabled", True))
        self.dr_friction_range: Tuple[float, float] = tuple(dr_cfg.get("friction_range", (0.6, 1.4)))
        self.dr_mass_scale_range: Tuple[float, float] = tuple(dr_cfg.get("mass_scale_range", (0.9, 1.1)))
        self.dr_motor_strength_range: Tuple[float, float] = tuple(dr_cfg.get("motor_strength_range", (0.85, 1.15)))
        self.dr_push_interval: int = int(dr_cfg.get("push_interval", 150))
        self.dr_push_magnitude: float = float(dr_cfg.get("push_magnitude", 8.0)) # Newtons
        self.realism = str(self.config.get("realism", "full"))
        self.push_enabled = bool(self.config.get("push_enabled", self.dr_enabled))
        if self.realism == "easy":
            self.dr_enabled = False
            self.push_enabled = False
            self.enable_delay = False
            self.enable_lag = False
            self.enable_backlash = False
        elif self.realism == "medium":
            self.dr_enabled = True
            self.push_enabled = False
            self.enable_delay = True
            self.delay_range = (0.010, 0.010)
            self.enable_lag = True
            self.enable_backlash = False
            self.dr_motor_strength_range = (1.0, 1.0)

        # Termination thresholds
        term_cfg = self.config.get("termination", {})
        self.min_torso_height: float = float(term_cfg.get("min_torso_height", 0.15))
        self.max_torso_height: float = float(term_cfg.get("max_torso_height", 0.45))
        self.min_upright_cosine: float = float(term_cfg.get("min_upright_cosine", 0.65))

        # Goal sampling ranges [vx, vy, wz]
        goal_cfg = self.config.get("goal_ranges", {})
        self.goal_vx_range: Tuple[float, float] = tuple(goal_cfg.get("vx", (-0.2, 0.6)))
        self.goal_vy_range: Tuple[float, float] = tuple(goal_cfg.get("vy", (-0.2, 0.2)))
        self.goal_wz_range: Tuple[float, float] = tuple(goal_cfg.get("wz", (-0.5, 0.5)))
        curriculum_cfg = self.config.get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.p_zero_cmd = float(curriculum_cfg.get("p_zero_cmd", 0.25))
        self.curriculum_scale = 1.0
        self.curriculum_initial_ranges = {
            "vx": tuple(curriculum_cfg.get("initial_vx", self.goal_vx_range)),
            "vy": tuple(curriculum_cfg.get("initial_vy", self.goal_vy_range)),
            "wz": tuple(curriculum_cfg.get("initial_wz", self.goal_wz_range)),
        }

        # Define Observation & Action spaces
        # Proprioception: joint pos (12) + joint vel (12) + base ang vel (3) + proj gravity (3) + prev action (12) = 42
        self.obs_dim: int = 12 + 12 + 3 + 3 + 12
        self.goal_dim: int = 3 + (len(self.budget_cost_names) if self.budget_goal_enabled else 0)
        reward_cfg = self.config.get("reward", {})
        self.reward_mode = str(reward_cfg.get("mode", "track"))
        self.sigma_v = float(reward_cfg.get("sigma_v", 0.15))
        self.sigma_w = float(reward_cfg.get("sigma_w", 0.30))
        self.fall_penalty = float(reward_cfg.get("fall_penalty", 0.0))
        self.budget_compliance_threshold = float(
            budget_cfg.get("compliance_threshold", 0.9)
        )
        self.stand_safe_weights = {
            "pose": float(reward_cfg.get("stand_safe_pose_weight", 5.0)),
            "symmetry": float(reward_cfg.get("stand_safe_symmetry_weight", 1.0)),
            "torque": float(reward_cfg.get("stand_safe_torque_weight", 2.0)),
            "action": float(reward_cfg.get("stand_safe_action_weight", 1.0)),
        }
        shaping_cfg = self.config.get("shaping", {})
        self.gait_shaping = bool(shaping_cfg.get("gait_shaping", False))
        self.action_filter_enabled = bool(shaping_cfg.get("action_filter", False))
        self.prev_action_obs_enabled = bool(shaping_cfg.get("prev_action_obs", False))
        self.smooth_penalties = bool(shaping_cfg.get("smooth_penalties", False))
        self.reset_velocity_enabled = bool(shaping_cfg.get("reset_velocity", False))
        self.action_filter_alpha = float(shaping_cfg.get("action_filter_alpha", 0.6))
        self.progress_weight = float(shaping_cfg.get("progress_weight", 0.0))
        self.air_time_weight = float(shaping_cfg.get("air_time_weight", 3.0))
        self.single_support_weight = float(shaping_cfg.get("single_support_weight", 0.05))
        self.swing_clearance_weight = float(shaping_cfg.get("swing_clearance_weight", 0.1))
        self.flight_penalty_weight = float(shaping_cfg.get("flight_penalty_weight", -0.5))
        self.zero_touchdown_weight = float(shaping_cfg.get("zero_touchdown_weight", -0.3))
        self.action_rate_weight = float(shaping_cfg.get("action_rate_weight", 0.02))
        self.action_second_diff_weight = float(
            shaping_cfg.get("action_second_diff_weight", 0.002)
        )
        self.joint_accel_weight = float(
            shaping_cfg.get("joint_accel_weight", 0.0)
        )
        self.posture_weight = float(shaping_cfg.get("posture_weight", 0.0))
        self.target_torso_height = float(
            shaping_cfg.get("target_torso_height", 0.25)
        )
        self.base_motion_weight = float(shaping_cfg.get("base_motion_weight", 0.05))
        self.smoothness_ramp = float(shaping_cfg.get("smoothness_ramp", 1.0))
        self.reset_velocity_probability = float(
            shaping_cfg.get("reset_velocity_probability", 0.5)
        )
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # Internal state buffers
        self._step_count: int = 0
        self._prev_action: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._prev_prev_action: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._action_filter_state: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._prev_joint_vel: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._filtered_action: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._delayed_action: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._backlash_state: np.ndarray = np.zeros(self.num_joints, dtype=np.float32)
        self._action_buffer: deque[Tuple[float, np.ndarray]] = deque(maxlen=20)
        
        self._active_delay: float = self.nominal_delay
        self._motor_strength_scale: np.ndarray = np.ones(self.num_joints, dtype=np.float32)
        
        self._desired_goal: np.ndarray = np.zeros(self.goal_dim, dtype=np.float32)
        self._push_force: np.ndarray = np.zeros(3, dtype=np.float32)
        self._push_steps_remaining: int = 0

        # Step timing & symmetry tracking
        self._foot_contact_times: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_foot_contact: np.ndarray = np.zeros(2, dtype=bool)
        self._foot_air_times: np.ndarray = np.zeros(2, dtype=np.float32)
        self._swing_peak_heights: np.ndarray = np.zeros(2, dtype=np.float32)

        # Baseline body mass backup for randomization
        self._nominal_torso_mass: float = float(self.model.body_mass[self.torso_body_id])
        self._nominal_geom_friction: np.ndarray = self.model.geom_friction.copy()

        self.adaptive_reward_fn = None
        self._viewer = None
        self._renderer = None
        if self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    # ----------------------------------------------------------------------
    # Core Gym API
    # ----------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Apply Domain Randomization if enabled
        if self.dr_enabled:
            self._apply_domain_randomization()
        else:
            self._active_delay = self.nominal_delay
            self._motor_strength_scale.fill(1.0)
            self.model.geom_friction[:] = self._nominal_geom_friction
            self.model.body_mass[self.torso_body_id] = self._nominal_torso_mass

        # Initialize robot pose around nominal default pose with small jitter
        qpos = self.model.qpos0.copy()
        qpos[0] = self.model.qpos0[0] + float(self.np_random.uniform(-0.005, 0.005))
        qpos[1] = self.model.qpos0[1] + float(self.np_random.uniform(-0.005, 0.005))
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # upright quat
        # Actuated joints
        joint_jitter = self.np_random.uniform(-0.02, 0.02, size=self.num_joints)
        for i, adr in enumerate(self.joint_qpos_adrs):
            qpos[adr] = self.default_qpos[i] + joint_jitter[i]
        
        qvel = np.zeros(self.model.nv, dtype=np.float32)
        vel_jitter = self.np_random.uniform(-0.01, 0.01, size=self.num_joints)
        for i, adr in enumerate(self.joint_dof_adrs):
            qvel[adr] = vel_jitter[i]

        qpos[2] = 0.0
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        qpos[2] = self._root_height_for_floor_contact()
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        # Reset state tracking buffers
        self._step_count = 0
        self._prev_action.fill(0.0)
        self._prev_prev_action.fill(0.0)
        self._action_filter_state.fill(0.0)
        self._prev_joint_vel.fill(0.0)
        self._filtered_action = self.default_qpos.copy()
        self._delayed_action = self.default_qpos.copy()
        self._backlash_state = self.default_qpos.copy()
        self._action_buffer.clear()
        
        # Prefill action buffer
        for i in range(10):
            t_hist = - (10 - i) * self.dt
            self._action_buffer.append((t_hist, np.zeros(self.num_joints, dtype=np.float32)))

        self.velocity_window.clear()
        for values in self.cost_windows.values():
            values.clear()
        self._foot_contact_times.fill(0.0)
        self._last_foot_contact.fill(False)
        self._foot_air_times.fill(0.0)
        self._swing_peak_heights.fill(0.0)
        self._push_force.fill(0.0)
        self._push_steps_remaining = 0

        # Sample or set desired goal [vx*, vy*, wz*] plus optional budgets.
        if options and "desired_goal" in options:
            self._desired_goal = np.array(options["desired_goal"], dtype=np.float32)
        else:
            self._desired_goal = self._sample_goal()
        if self._desired_goal.shape != (self.goal_dim,):
            raise ValueError(f"desired_goal must have shape {(self.goal_dim,)}")
        if (
            self.reset_velocity_enabled
            and abs(float(self._desired_goal[0])) > 0.1
            and self.np_random.random() < self.reset_velocity_probability
        ):
            self.data.qvel[0] = float(
                self.np_random.uniform(0.5, 1.0) * self._desired_goal[0]
            )
            mujoco.mj_forward(self.model, self.data)

        obs_dict = self._get_obs_dict()
        info = self._get_info(action=np.zeros(self.num_joints, dtype=np.float32))

        return obs_dict, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[Dict[str, Any], List[Dict[str, Any]], Tuple[Dict[str, Any], ...], None] = None,
    ) -> Union[float, np.ndarray]:
        """Compute reward through the shared vectorized reward module."""
        is_batch = np.asarray(achieved_goal).ndim > 1
        res = compute_reward_batch(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            info=info,
            reward_mode=self.reward_mode,
            adaptive_reward_fn=self.adaptive_reward_fn,
            budget_goal_enabled=self.budget_goal_enabled,
            fall_penalty=self.fall_penalty,
            sigma_v=self.sigma_v,
            sigma_w=self.sigma_w,
            stand_safe_weights=getattr(self, "stand_safe_weights", None),
            progress_weight=self.progress_weight,
            gait_shaping=self.gait_shaping,
            smooth_penalties=self.smooth_penalties,
        )
        return res if is_batch else float(res[0])

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_state = action
        if self.action_filter_enabled:
            alpha = float(np.clip(self.action_filter_alpha, 0.0, 1.0))
            self._action_filter_state = (
                alpha * action + (1.0 - alpha) * self._action_filter_state
            ).astype(np.float32)
            action_state = self._action_filter_state
        curr_time = self._step_count * self.dt
        self._action_buffer.append((curr_time, action_state.copy()))

        # Actuator realistic pipeline:
        # 1. Action latency buffer (delay)
        if self.enable_delay:
            delayed_norm_action = self._retrieve_delayed_action(curr_time, self._active_delay)
        else:
            delayed_norm_action = action_state.copy()

        # 2. Map normalized actions asymmetrically from the default pose to limits.
        target_positions = self._action_to_target(delayed_norm_action)

        # 3. First-order servo lag filter
        if self.enable_lag:
            alpha = self.dt / (self.servo_lag_tau + self.dt)
            self._filtered_action = self._filtered_action + alpha * (target_positions - self._filtered_action)
        else:
            self._filtered_action = target_positions.copy()

        # 4. Mechanical backlash / deadband
        if self.enable_backlash:
            delta = self._filtered_action - self._backlash_state
            mask = np.abs(delta) > self.backlash_deadband
            self._backlash_state[mask] = self._backlash_state[mask] + np.sign(delta[mask]) * (
                np.abs(delta[mask]) - self.backlash_deadband
            )
            actuator_cmd = self._backlash_state.copy()
        else:
            actuator_cmd = self._filtered_action.copy()

        # Apply motor strength scaling (simulating battery degradation / servo variation)
        applied_ctrl = self.default_qpos + (actuator_cmd - self.default_qpos) * self._motor_strength_scale
        applied_ctrl = np.clip(applied_ctrl, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[self.actuator_ids] = applied_ctrl

        # Random horizontal push perturbation
        if self.push_enabled and (self._step_count % self.dr_push_interval == 0) and self._step_count > 0:
            push_ang = self.np_random.uniform(0, 2 * np.pi)
            self._push_force = np.array([
                self.dr_push_magnitude * np.cos(push_ang),
                self.dr_push_magnitude * np.sin(push_ang),
                0.0
            ], dtype=np.float32)
            self._push_steps_remaining = 5  # active for 5 steps (0.1s)

        # Step MuJoCo physics with frame skip
        for _ in range(self.frame_skip):
            if self._push_steps_remaining > 0:
                self.data.xfrc_applied[self.torso_body_id, :3] = self._push_force
                self._push_steps_remaining -= 1
            else:
                self.data.xfrc_applied[self.torso_body_id, :3] = 0.0

            mujoco.mj_step(self.model, self.data)

        # Update sliding window for achieved goal
        instant_body_vel = self._get_body_frame_velocity()
        self.velocity_window.append(instant_body_vel)

        # Compute observations, terminations, and step metrics
        obs_dict = self._get_obs_dict()
        info = self._get_info(action=action_state)
        
        terminated = self._is_terminated()
        self._step_count += 1
        truncated = (not terminated) and (self._step_count >= self.max_episode_steps)

        info["fell"] = bool(terminated)
        reward = float(self.compute_reward(obs_dict["achieved_goal"], obs_dict["desired_goal"], info))

        self._prev_prev_action = self._prev_action.copy()
        self._prev_action = action_state.copy()

        if self.render_mode == "human":
            self.render()

        return obs_dict, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current MuJoCo state for headless video generation."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        camera = self._renderer.scene.camera[0]
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.20
        camera.azimuth = 0.0
        camera.elevation = -10.0
        camera.lookat[:] = self.data.xpos[self.torso_body_id]
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    # ----------------------------------------------------------------------
    # Helper & Sensor Calculations
    # ----------------------------------------------------------------------
    def _sample_goal(self) -> np.ndarray:
        """Sample velocity goals and, when enabled, independent log-uniform budgets."""
        ranges = self._current_goal_ranges()
        if self.np_random.random() < self.p_zero_cmd:
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            vx = self.np_random.uniform(*ranges["vx"])
            vy = self.np_random.uniform(*ranges["vy"])
            wz = self.np_random.uniform(*ranges["wz"])
        goal = [vx, vy, wz]
        if self.budget_goal_enabled:
            if self.np_random.random() < float(
                self.config.get("budget_goal", {}).get("p_all_unconstrained", 0.2)
            ):
                budgets = np.ones(len(self.budget_cost_names), dtype=np.float32)
            else:
                upper = np.ones(len(self.budget_cost_names), dtype=np.float32)
                budgets = np.exp(
                    self.np_random.uniform(np.log(self.budget_min), np.log(upper))
                )
            goal.extend(budgets.tolist())
        return np.asarray(goal, dtype=np.float32)

    def _current_goal_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges: Dict[str, Tuple[float, float]] = {}
        for name, full_range in (("vx", self.goal_vx_range), ("vy", self.goal_vy_range), ("wz", self.goal_wz_range)):
            initial = np.asarray(self.curriculum_initial_ranges[name], dtype=np.float32)
            scaled = np.clip(initial * self.curriculum_scale, min(full_range), max(full_range))
            ranges[name] = (float(scaled[0]), float(scaled[1]))
        return ranges

    def set_curriculum_scale(self, scale: float) -> None:
        self.curriculum_scale = max(0.0, float(scale))

    def set_smoothness_ramp(self, ramp: float) -> None:
        self.smoothness_ramp = float(np.clip(ramp, 0.0, 1.0))

    def _action_to_target(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        positive_span = self.jnt_max - self.default_qpos
        negative_span = self.default_qpos - self.jnt_min
        target = np.where(
            action >= 0.0,
            self.default_qpos + action * self.action_range_fraction * positive_span,
            self.default_qpos + action * self.action_range_fraction * negative_span,
        )
        return np.clip(target, self.ctrl_min, self.ctrl_max).astype(np.float32)

    def _tracking_reward_batch(self, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
        v_diff = achieved[:, :2] - desired[:, :2]
        w_diff = achieved[:, 2] - desired[:, 2]
        v_err_sq = np.sum(v_diff ** 2, axis=-1)
        w_err_sq = w_diff ** 2
        return (
            np.exp(-v_err_sq / (2.0 * self.sigma_v ** 2))
            * np.exp(-w_err_sq / (2.0 * self.sigma_w ** 2))
        ).astype(np.float32)

    def _root_height_for_floor_contact(self) -> float:
        lowest = float("inf")
        for geom_id in self.foot_geom_ids:
            mesh_id = int(self.model.geom_dataid[geom_id])
            start = int(self.model.mesh_vertadr[mesh_id])
            count = int(self.model.mesh_vertnum[mesh_id])
            vertices = self.model.mesh_vert[start:start + count]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            lowest = min(
                lowest,
                float(np.min(self.data.geom_xpos[geom_id, 2] + vertices @ rotation[2, :])),
            )
        return float(self.data.qpos[2] - lowest)

    def set_desired_goal(self, goal: np.ndarray) -> None:
        """Explicitly set desired goal velocity (used in curriculum and evaluation)."""
        self._desired_goal = np.asarray(goal, dtype=np.float32)

    def _retrieve_delayed_action(self, current_time: float, delay_sec: float) -> np.ndarray:
        """Interpolate action from history buffer at (current_time - delay_sec)."""
        target_t = current_time - delay_sec
        if len(self._action_buffer) == 0:
            return np.zeros(self.num_joints, dtype=np.float32)

        # Find neighboring frames
        buf = list(self._action_buffer)
        if target_t <= buf[0][0]:
            return buf[0][1].copy()
        if target_t >= buf[-1][0]:
            return buf[-1][1].copy()

        for i in range(len(buf) - 1):
            t0, a0 = buf[i]
            t1, a1 = buf[i + 1]
            if t0 <= target_t <= t1:
                ratio = (target_t - t0) / max(1e-6, t1 - t0)
                return (1.0 - ratio) * a0 + ratio * a1

        return buf[-1][1].copy()

    def _get_body_frame_velocity(self) -> np.ndarray:
        """Compute base linear velocity (vx, vy) and yaw rate (wz) in base body frame."""
        # World linear velocity from qvel[0:3]
        v_world = self.data.qvel[0:3]
        # Torso rotation matrix
        quat = self.data.qpos[3:7]  # w, x, y, z
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        rot_mat = mat.reshape(3, 3)

        # Body-frame linear velocity: R^T * v_world
        v_body = rot_mat.T @ v_world
        vx_b = float(v_body[0])
        vy_b = float(v_body[1])

        # Angular velocity: qvel[3:6] in MuJoCo freejoint is body coordinates
        wz_b = float(self.data.qvel[5])

        return np.array([vx_b, vy_b, wz_b], dtype=np.float32)

    def _get_achieved_goal(self) -> np.ndarray:
        """Compute velocity and optional normalized windowed safety costs."""
        if len(self.velocity_window) == 0:
            velocity = self._get_body_frame_velocity()
        else:
            velocity = np.mean(np.array(self.velocity_window), axis=0).astype(np.float32)
        if not self.budget_goal_enabled:
            return velocity
        costs = []
        for name, d_max in zip(self.budget_cost_names, self.budget_d_max):
            values = self.cost_windows[name]
            mean_cost = float(np.mean(values)) if values else 0.0
            costs.append(np.clip(mean_cost / max(float(d_max), 1e-6), 0.0, 1.0))
        return np.concatenate([velocity, np.asarray(costs, dtype=np.float32)])

    def _get_projected_gravity(self) -> np.ndarray:
        """Project world gravity [0, 0, -1] into robot base frame."""
        quat = self.data.qpos[3:7]
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        rot_mat = mat.reshape(3, 3)
        # R^T * [0, 0, -1] is the negative 3rd column of R
        return - rot_mat[2, :].astype(np.float32)

    def _upright_cosine(self) -> float:
        """Cos(tilt angle) between robot base z-axis and world vertical z-axis."""
        quat = self.data.qpos[3:7]
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return float(mat[8])  # rot_mat[2, 2]

    def _feet_contact(self) -> np.ndarray:
        """Return binary contact status for [right_foot, left_foot]."""
        contacts = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for foot_idx, fg_id in enumerate(self.foot_geom_ids):
                if (c.geom1 == fg_id and c.geom2 == self.floor_geom_id) or (
                    c.geom2 == fg_id and c.geom1 == self.floor_geom_id
                ):
                    contacts[foot_idx] = 1.0
        return contacts

    def _feet_tangential_slip(self) -> float:
        """Compute tangential slip velocity (m/s) of feet in contact with ground."""
        contacts = self._feet_contact()
        total_slip = 0.0
        for foot_idx, in_contact in enumerate(contacts):
            if in_contact > 0.5:
                body_id = self.foot_body_ids[foot_idx]
                # Body velocity (6-dim: 3 ang, 3 lin in world frame)
                v_lin = self.data.cvel[body_id, 3:6]
                slip_speed = float(np.linalg.norm(v_lin[:2]))
                total_slip += slip_speed
        return float(total_slip)

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Construct full observation dictionary."""
        # Joint positions relative to default
        joint_pos = np.array([
            self.data.qpos[adr] - self.default_qpos[i]
            for i, adr in enumerate(self.joint_qpos_adrs)
        ], dtype=np.float32)
        # Joint velocities
        joint_vel = np.array([
            self.data.qvel[adr]
            for adr in self.joint_dof_adrs
        ], dtype=np.float32)
        # Base angular velocity in body frame
        base_ang_vel = self.data.qvel[3:6].astype(np.float32)
        # Projected gravity vector
        proj_gravity = self._get_projected_gravity()
        # Previous action
        prev_act = (
            self._action_filter_state.copy()
            if self.action_filter_enabled and self.prev_action_obs_enabled
            else self._prev_action.copy()
        )

        proprio = np.concatenate([
            joint_pos * self.obs_scale["joint_pos"],
            joint_vel * self.obs_scale["joint_vel"],
            base_ang_vel * self.obs_scale["base_ang_vel"],
            proj_gravity * self.obs_scale["proj_gravity"],
            prev_act * self.obs_scale["prev_action"],
        ]).astype(np.float32)

        achieved_goal = self._get_achieved_goal()
        desired_goal = self._desired_goal.copy()

        return {
            "observation": proprio,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def _get_info(self, action: np.ndarray) -> Dict[str, Any]:
        """Compute per-step metrics and constraint violations for adaptive reward."""
        # 1. Torque saturation ratio: actuators producing >= 98% of peak torque
        torques = np.abs(self.data.actuator_force[self.actuator_ids])
        sat_mask = torques >= (0.98 * self.torque_limit)
        torque_saturation_ratio = float(np.mean(sat_mask))
        torque_rms = float(np.sqrt(np.mean((torques / self.torque_limit) ** 2)))

        # 2. Torso tilt violation: (1 - cos(theta))
        upright = self._upright_cosine()
        torso_tilt = float(max(0.0, 1.0 - upright))

        # 3. Foot slip: tangential velocity during contact
        foot_slip = float(self._feet_tangential_slip())

        # 4. Action rate: magnitude of consecutive action differences
        action_rate = float(np.mean((action - self._prev_action) ** 2))
        action_second_diff = float(
            np.mean((action - 2.0 * self._prev_action + self._prev_prev_action) ** 2)
        )
        action_saturation_ratio = float(np.mean(np.abs(action) > 0.95))
        if self.budget_goal_enabled:
            instantaneous_costs = {
                "torque_rms": torque_rms,
                "torso_tilt": torso_tilt,
                "foot_slip": foot_slip,
                "action_rate": action_rate,
            }
            for name in self.budget_cost_names:
                self.cost_windows[name].append(float(instantaneous_costs[name]))

        achieved = self._get_achieved_goal()
        desired = self._desired_goal
        vel_err_xy = np.linalg.norm(achieved[:2] - desired[:2])
        wz_err = abs(achieved[2] - desired[2])
        r_track = float(self._tracking_reward_batch(achieved[None, :3], desired[None, :3])[0])

        # 5. Joint limit proximity penalty
        qpos_j = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adrs], dtype=np.float32)
        pose_deviation = float(np.mean((qpos_j - self.default_qpos) ** 2))
        stand_symmetry = float(
            np.mean(np.abs(qpos_j[: self.num_joints // 2] + qpos_j[self.num_joints // 2 :]))
        )
        dist_min = qpos_j - self.jnt_min
        dist_max = self.jnt_max - qpos_j
        min_margin = np.minimum(dist_min, dist_max)
        joint_limit_proximity = float(np.mean(np.clip(0.10 - min_margin, 0.0, None) ** 2))

        # 6. Foot clearance: penalize low swing foot height
        contacts = self._feet_contact()
        touchdown = contacts > 0.5
        touchdown_events = touchdown & ~self._last_foot_contact
        raw_air_time = self._foot_air_times.copy()
        for idx in range(2):
            if touchdown[idx]:
                self._foot_air_times[idx] = 0.0
                self._swing_peak_heights[idx] = 0.0
            else:
                self._foot_air_times[idx] += self.dt
                self._swing_peak_heights[idx] = max(
                    self._swing_peak_heights[idx],
                    float(self.data.xpos[self.foot_body_ids[idx], 2]),
                )
        self._last_foot_contact = touchdown.copy()
        foot_clearance_penalty = 0.0
        for idx, in_contact in enumerate(contacts):
            if in_contact < 0.5:  # foot in swing phase
                foot_z = float(self.data.xpos[self.foot_body_ids[idx], 2])
                target_clearance = 0.04  # 4 cm target swing clearance
                if foot_z < target_clearance:
                    foot_clearance_penalty += (target_clearance - foot_z)
        foot_clearance = float(foot_clearance_penalty)
        touchdown_air_reward = float(
            np.sum(np.clip(raw_air_time[touchdown_events] - 0.25, -0.1, 0.15))
            * self.air_time_weight
        )
        single_support = float(np.sum(contacts) == 1) * self.single_support_weight
        clearance_reward = float(
            sum(
                min(float(height) / 0.05, 1.0)
                for height in self._swing_peak_heights
                if height > 0.0
            )
            * self.swing_clearance_weight
        )
        r_gait_move = touchdown_air_reward + single_support + clearance_reward
        r_gait_still = float(np.sum(touchdown_events)) * self.zero_touchdown_weight
        r_flight = float(np.sum(contacts) == 0) * self.flight_penalty_weight

        # 7. Energy: mechanical work sum(|tau * qvel|) * dt
        qvel_j = np.array([self.data.qvel[adr] for adr in self.joint_dof_adrs], dtype=np.float32)
        joint_accel = float(np.mean(((qvel_j - self._prev_joint_vel) / self.dt) ** 2))
        self._prev_joint_vel = qvel_j.copy()
        energy = float(np.sum(torques * np.abs(qvel_j)) * self.dt)

        # 8. Gait symmetry tracking: contact duration ratio
        for idx in range(2):
            if contacts[idx] > 0.5:
                self._foot_contact_times[idx] += self.dt
        t_r, t_l = self._foot_contact_times[0], self._foot_contact_times[1]
        gait_asymmetry = float(abs(t_r - t_l) / max(0.1, t_r + t_l))

        torso_z = float(self.data.qpos[2])
        r_smooth = 0.0
        if self.smooth_penalties:
            ramp = max(0.0, min(1.0, self.smoothness_ramp))
            r_smooth = -ramp * (
                self.action_rate_weight * action_rate
                + self.action_second_diff_weight * action_second_diff
                + self.base_motion_weight
                * (
                    float(self.data.qvel[2]) ** 2
                    + float(np.sum(self.data.qvel[3:5] ** 2))
                )
                + self.joint_accel_weight * joint_accel
                + self.posture_weight
                * (
                    (torso_z - self.target_torso_height) ** 2
                    + (1.0 - upright) ** 2
                )
            )

        return {
            "r_track": r_track,
            "torque_saturation_ratio": torque_saturation_ratio,
            "torque_rms": torque_rms,
            "torso_tilt": torso_tilt,
            "foot_slip": foot_slip,
            "action_rate": action_rate,
            "action_saturation_ratio": action_saturation_ratio,
            "pose_deviation": pose_deviation,
            "stand_symmetry": stand_symmetry,
            "joint_limit_proximity": joint_limit_proximity,
            "foot_clearance": foot_clearance,
            "energy": energy,
            "gait_asymmetry": gait_asymmetry,
            "torso_height": torso_z,
            "upright_cosine": upright,
            "vel_err_xy": float(vel_err_xy),
            "wz_err": float(wz_err),
            "action_second_diff": action_second_diff,
            "joint_acceleration": joint_accel,
            "r_smooth": float(r_smooth),
            "r_gait_move": float(r_gait_move),
            "r_gait_still": float(r_gait_still),
            "r_flight": float(r_flight),
            "touchdown_events": touchdown_events.astype(np.float32),
            "air_time": raw_air_time,
            "achieved_goal": achieved,
            "desired_goal": desired,
            "fell": False,
        }

    def _is_terminated(self) -> bool:
        """Check termination: torso height fall or excessive tilt."""
        torso_z = float(self.data.qpos[2])
        if torso_z < self.min_torso_height or torso_z > self.max_torso_height:
            return True

        upright = self._upright_cosine()
        if upright < self.min_upright_cosine:
            return True

        return False

    def _apply_domain_randomization(self) -> None:
        """Randomize ground friction, torso mass, action latency, and motor strength."""
        # Friction
        fric_scale = self.np_random.uniform(self.dr_friction_range[0], self.dr_friction_range[1])
        self.model.geom_friction[self.floor_geom_id, 0] = self._nominal_geom_friction[self.floor_geom_id, 0] * fric_scale

        # Torso mass
        mass_scale = self.np_random.uniform(self.dr_mass_scale_range[0], self.dr_mass_scale_range[1])
        self.model.body_mass[self.torso_body_id] = self._nominal_torso_mass * mass_scale

        # Action latency delay
        self._active_delay = self.np_random.uniform(self.delay_range[0], self.delay_range[1])

        # Motor strength scale
        self._motor_strength_scale = self.np_random.uniform(
            self.dr_motor_strength_range[0], self.dr_motor_strength_range[1], size=self.num_joints
        ).astype(np.float32)

    def render(self):
        """Render frame."""
        if self.render_mode == "rgb_array":
            if self._viewer is None:
                self._viewer = mujoco.Renderer(self.model, height=480, width=640)
            self._viewer.update_scene(self.data, camera="side")
            return self._viewer.render()
        elif self.render_mode == "human":
            # For headless or local display
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer = None
