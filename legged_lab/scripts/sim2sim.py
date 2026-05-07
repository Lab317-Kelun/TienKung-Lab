#!/usr/bin/env python3
# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import os
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard


# Robot1_6 joint ordering in Isaac Lab (policy/training ordering).
ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


class SimToSimCfg:
    """Sim2Sim configuration for robot1_6.

    Must be kept consistent with the training configuration:
    - `legged_lab/envs/robot1_6/robot1_6_env.py::compute_current_observations`
    - `legged_lab/envs/robot1_6/walk_cfg.py::Robot1_6WalkFlatEnvCfg`
    - `legged_lab/assets/robot1_6/robot1_6.py::ROBOT1_6_CFG.actuators` (PD gains)
    """

    class sim:
        sim_duration = 100.0
        dt = 0.005
        decimation = 4

        num_action = 12  # robot1_6 has 12 joints (lower body only)
        num_obs_per_step = 51  # 3(ang_vel) + 3(gravity) + 3(command) + 12(joint_pos) + 12(joint_vel) + 12(action) + 2(sin) + 2(cos) + 2(phase_ratio) = 51
        actor_obs_history_length = 10

        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

        # Optional fixed action delay (in control steps). Training randomizes in [0, 1].
        action_delay_steps = 0

        # Observation scales (must match walk_cfg.py)
        obs_scales = {
            "ang_vel": 1.0,
            "projected_gravity": 1.0,
            "commands": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "actions": 1.0,
        }

    class robot:
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.80

    class commands:
        lin_vel_x_range = (-2.0, 2.0)
        lin_vel_y_range = (-2.0, 2.0)
        ang_vel_z_range = (-2.0, 2.0)
        step = 0.1


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = float(self.cfg.sim.dt)

        self.policy = _load_policy(policy_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        
        # ✅ 按照 Q1 的方式分离初始化步骤
        self._init_mappings_and_indices()
        self._init_gains_and_limits()
        self._init_buffers()
        self._init_robot_state()

    def _init_mappings_and_indices(self) -> None:
        if self.model.nu != self.cfg.sim.num_action:
            raise ValueError(f"MuJoCo model actuator num mismatch: expected {self.cfg.sim.num_action}, got {self.model.nu}.")

        mujoco_actuator_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)
        ]
        missing = [name for name in ISAAC_JOINT_NAMES if name not in mujoco_actuator_names]
        if missing:
            raise ValueError(f"MuJoCo model missing actuators for joints: {missing}")

        mujoco_name_to_idx = {name: idx for idx, name in enumerate(mujoco_actuator_names)}
        self.mujoco_to_isaac_idx = [mujoco_name_to_idx[name] for name in ISAAC_JOINT_NAMES]
        self.isaac_to_mujoco_idx = np.argsort(self.mujoco_to_isaac_idx).tolist()

        # Cache qpos/qvel indices for actuated joints in MuJoCo actuator order.
        self._qpos_idx = np.zeros(self.model.nu, dtype=np.int32)
        self._qvel_idx = np.zeros(self.model.nu, dtype=np.int32)
        for act_id, joint_name in enumerate(mujoco_actuator_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Actuator '{joint_name}' does not map to a joint with the same name.")
            self._qpos_idx[act_id] = int(self.model.jnt_qposadr[joint_id])
            self._qvel_idx[act_id] = int(self.model.jnt_dofadr[joint_id])
    
    def _init_gains_and_limits(self) -> None:        
        kp_by_name: dict[str, float] = {}
        kd_by_name: dict[str, float] = {}

        # legs - using Unitree G1 values
        for name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
            kp_by_name[name] = 70  # 40.18
            kd_by_name[name] = 5.0  # 2.56
        for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
            kp_by_name[name] = 140  # 99.10
            kd_by_name[name] = 8.0  # 6.31
        for name in ("left_hip_yaw_joint", "right_hip_yaw_joint"):
            kp_by_name[name] = 45  # 40.18
            kd_by_name[name] = 3.0  # 2.56
        for name in ("left_knee_joint", "right_knee_joint"):
            kp_by_name[name] = 140  # 99.10
            kd_by_name[name] = 8.0  # 6.31

        # feet - using Unitree G1 values (2x STIFFNESS_5020 and DAMPING_5020)
        for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            kp_by_name[name] = 34
            kd_by_name[name] = 2.4
        for name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
            kp_by_name[name] = 24
            kd_by_name[name] = 1.65

        mujoco_actuator_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)
        ]
        self.kp = np.array([kp_by_name[name] for name in mujoco_actuator_names], dtype=np.float64)
        self.kd = np.array([kd_by_name[name] for name in mujoco_actuator_names], dtype=np.float64)

        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
    
    def _init_buffers(self) -> None:
        self.dt = float(self.cfg.sim.dt)
        self.control_dt = float(self.cfg.sim.decimation * self.cfg.sim.dt)

        self.dof_pos_mujoco = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self.dof_vel_mujoco = np.zeros(self.cfg.sim.num_action, dtype=np.float64)

        # robot1_6 default joint positions (from robot1_6.py init_state.joint_pos) in Isaac Lab order.
        # Order matches ISAAC_JOINT_NAMES: [left_hip_pitch, right_hip_pitch, left_hip_roll, right_hip_roll,
        #                                    left_hip_yaw, right_hip_yaw, left_knee, right_knee,
        #                                    left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll]
        self.default_dof_pos_isaac = np.array(
            [-0.20, 0.20, 0.0, 0.0, -0.05, 0.05, 0.32, -0.32, -0.15, 0.15, 0.0, 0.0],
            dtype=np.float64
        )
        self.default_dof_pos_mujoco = self.default_dof_pos_isaac[self.isaac_to_mujoco_idx]

        # Action stored in Isaac joint order. Observation uses last applied action.
        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self._action_cmd_buf = [np.zeros(self.cfg.sim.num_action, dtype=np.float64) for _ in range(2)]

        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

        # Gait parameters
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
        
        # Torque recording for plotting
        self.torque_history = []  # List of torque arrays (in Isaac order)
        self.time_history = []  # List of time stamps
        
        # Real-time plotting setup
        self.plot_update_interval = 10  # Update plot every N control steps
        self.plot_counter = 0
        self.fig = None
        self.axes = None
        self.lines = {}  # Store line objects for each joint
        self.peak_lines = {}  # Store peak limit lines

    def _init_robot_state(self) -> None:
        """Initialize robot state with default joint positions to prevent leg crossing."""
        # Set base position and orientation
        self.data.qpos[0:3] = [0.0, 0.0, 1.0]  # x, y, z
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (w, x, y, z)
        
        # Set initial joint positions in MuJoCo order
        self.data.qpos[self._qpos_idx] = self.default_dof_pos_mujoco
        
        # Set initial velocities to zero
        self.data.qvel[:] = 0.0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

    def _read_actuated_joint_state_mujoco_order(self) -> tuple[np.ndarray, np.ndarray]:
        self.dof_pos_mujoco = self.data.qpos[self._qpos_idx].astype(np.float64, copy=False)
        self.dof_vel_mujoco = self.data.qvel[self._qvel_idx].astype(np.float64, copy=False)
        return self.dof_pos_mujoco, self.dof_vel_mujoco

    def get_obs(self) -> np.ndarray:
        # Fallback: get world frame and convert to body frame.
        imu_angvel_world = self.data.sensor("imu_angvel").data.astype(np.float64)
        base_quat_wxyz = self.data.qpos[3:7].astype(np.float64)
        base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
        ang_vel = self.quat_rotate_inverse(base_quat_xyzw, imu_angvel_world)

        # Projected gravity in body frame.
        base_quat_wxyz = self.data.qpos[3:7].astype(np.float64)
        base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
        projected_gravity = self.quat_rotate_inverse(base_quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        
        # Joint states
        dof_pos_mujoco, dof_vel_mujoco = self._read_actuated_joint_state_mujoco_order()
        dof_pos_isaac = (dof_pos_mujoco - self.default_dof_pos_mujoco)[self.mujoco_to_isaac_idx]
        dof_vel_isaac = dof_vel_mujoco[self.mujoco_to_isaac_idx]
        
        obs = np.concatenate(
            [
                ang_vel * self.cfg.sim.obs_scales["ang_vel"],  # 3
                projected_gravity * self.cfg.sim.obs_scales["projected_gravity"],  # 3
                self.command_vel * self.cfg.sim.obs_scales["commands"],  # 3
                dof_pos_isaac * self.cfg.sim.obs_scales["joint_pos"],  # 12
                dof_vel_isaac * self.cfg.sim.obs_scales["joint_vel"],  # 12
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions) * self.cfg.sim.obs_scales["actions"],  # 12
                np.sin(2 * np.pi * self.gait_phase),  # 2 (gait phase info - robot1_6 specific)
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def _compute_target_pos_mujoco_order(self, action_isaac: np.ndarray) -> np.ndarray:
        action_scaled = action_isaac * self.cfg.sim.action_scale
        target_pos_isaac = action_scaled + self.default_dof_pos_isaac
        return target_pos_isaac[self.isaac_to_mujoco_idx]

    def _position_pd_torque(self, target_pos_mujoco: np.ndarray) -> np.ndarray:
        dof_pos_mujoco, dof_vel_mujoco = self._read_actuated_joint_state_mujoco_order()
        pos_err = target_pos_mujoco - dof_pos_mujoco
        vel_err = -dof_vel_mujoco
        torque = self.kp * pos_err + self.kd * vel_err
        return np.clip(torque, self.ctrl_lo, self.ctrl_hi)
        
    def run(self) -> None:
        self._setup_keyboard_listener()
        self.listener.start()
        
        # Initialize real-time plotting
        self._init_realtime_plot()

        # For velocity printing
        print_counter = 0
        print_interval = 50  # Print every 50 steps (50 * 4 * 0.005 = 1 second)

        while self.data.time < float(self.cfg.sim.sim_duration):
            obs_history = self.get_obs()

            action_np = self.policy(obs_history)
            action_np = np.array(action_np, dtype=np.float64).reshape(-1)
            action_cmd = np.clip(
                action_np[: self.cfg.sim.num_action], -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
            )

            # Fixed action delay (0 or 1 step) to match training domain randomization capability.
            delay = int(self.cfg.sim.action_delay_steps)
            if delay not in (0, 1):
                raise ValueError(f"action_delay_steps must be 0 or 1, got {delay}.")
            self._action_cmd_buf.pop(0)
            self._action_cmd_buf.append(action_cmd)
            self.action = self._action_cmd_buf[0 if delay == 1 else 1].copy()

            target_pos_mujoco = self._compute_target_pos_mujoco_order(self.action)
            
            for _ in range(int(self.cfg.sim.decimation)):
                step_start_time = time.time()
                torque_mujoco = self._position_pd_torque(target_pos_mujoco)
                self.data.ctrl[:] = torque_mujoco
                
                # Record torque in Isaac order for plotting
                torque_isaac = torque_mujoco[self.mujoco_to_isaac_idx]
                self.torque_history.append(torque_isaac.copy())
                self.time_history.append(self.data.time)
                
                mujoco.mj_step(self.model, self.data)
                self.viewer.render()
                
                elapsed = time.time() - step_start_time
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.episode_length_buf += 1
            self.calculate_gait_para()
            
            # Update real-time plot
            self.plot_counter += 1
            if self.plot_counter >= self.plot_update_interval:
                self.plot_counter = 0
                self._update_realtime_plot()
            
            # Print velocity comparison every print_interval steps
            print_counter += 1
            if print_counter >= print_interval:
                print_counter = 0
                # Get actual base velocity from MuJoCo
                base_lin_vel = self.data.sensor("imu_linvel").data.astype(np.double)
                base_ang_vel = self.data.sensor("imu_angvel").data.astype(np.double)
                # Get base_link height (z position)
                base_height = self.data.qpos[2]  # z position in qpos: [x, y, z, qw, qx, qy, qz, ...]
                
                print(f"\n[Time: {self.data.time:.2f}s]")
                print(f"命令速度 (Command): vx={self.command_vel[0]:+.3f} m/s, vy={self.command_vel[1]:+.3f} m/s, yaw={self.command_vel[2]:+.3f} rad/s")
                print(f"实际速度 (Actual):  vx={base_lin_vel[0]:+.3f} m/s, vy={base_lin_vel[1]:+.3f} m/s, yaw={base_ang_vel[2]:+.3f} rad/s")
                print(f"速度误差 (Error):   vx={base_lin_vel[0]-self.command_vel[0]:+.3f} m/s, vy={base_lin_vel[1]-self.command_vel[1]:+.3f} m/s, yaw={base_ang_vel[2]-self.command_vel[2]:+.3f} rad/s")
                print(f"Base Link 高度:     z={base_height:+.3f} m")

        self.listener.stop()
        self.viewer.close()
        
        # Plot torque curves
        self._plot_torque_curves()

    @staticmethod
    def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_w = q_xyzw[-1]
        q_vec = q_xyzw[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c

    def calculate_gait_para(self) -> None:
        """Update gait phase parameters based on simulation time and offset."""
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def _adjust_command_vel(self, idx: int, increment: float) -> None:
        self.command_vel[idx] += float(increment)
        if idx == 0:
            lo, hi = self.cfg.commands.lin_vel_x_range
        elif idx == 1:
            lo, hi = self.cfg.commands.lin_vel_y_range
        else:
            lo, hi = self.cfg.commands.ang_vel_z_range
        self.command_vel[idx] = float(np.clip(self.command_vel[idx], lo, hi))

    def _init_realtime_plot(self) -> None:
        """Initialize real-time torque plotting window."""
        plt.ion()  # Turn on interactive mode
        
        # Peak torque limits from XML file
        peak_limits = {
            "hip_pitch": 330.0,
            "hip_roll": 330.0,
            "hip_yaw": 115.0,
            "knee": 330.0,
            "ankle_pitch": 75.0,
            "ankle_roll": 75.0,
        }
        
        # Group joints by type
        self.joint_groups = {
            "Hip Pitch": {
                "indices": [0, 1],
                "names": ["Left Hip Pitch", "Right Hip Pitch"],
                "peak": peak_limits["hip_pitch"],
            },
            "Hip Roll": {
                "indices": [2, 3],
                "names": ["Left Hip Roll", "Right Hip Roll"],
                "peak": peak_limits["hip_roll"],
            },
            "Hip Yaw": {
                "indices": [4, 5],
                "names": ["Left Hip Yaw", "Right Hip Yaw"],
                "peak": peak_limits["hip_yaw"],
            },
            "Knee": {
                "indices": [6, 7],
                "names": ["Left Knee", "Right Knee"],
                "peak": peak_limits["knee"],
            },
            "Ankle Pitch": {
                "indices": [8, 9],
                "names": ["Left Ankle Pitch", "Right Ankle Pitch"],
                "peak": peak_limits["ankle_pitch"],
            },
            "Ankle Roll": {
                "indices": [10, 11],
                "names": ["Left Ankle Roll", "Right Ankle Roll"],
                "peak": peak_limits["ankle_roll"],
            },
        }
        
        # Create subplots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 12))
        self.fig.suptitle("Real-time Joint Torque Curves", fontsize=16, fontweight="bold")
        axes_flat = self.axes.flatten()
        
        # Initialize plots for each joint group
        for idx, (group_name, group_info) in enumerate(self.joint_groups.items()):
            ax = axes_flat[idx]
            self.lines[group_name] = []
            self.peak_lines[group_name] = []
            
            # Initialize lines for each joint
            for joint_name in group_info["names"]:
                line, = ax.plot([], [], label=joint_name, linewidth=1.5, alpha=0.8)
                self.lines[group_name].append(line)
            
            # Add peak limit lines
            peak = group_info["peak"]
            peak_line_pos, = ax.plot([], [], color="r", linestyle="--", linewidth=2, 
                                     label=f"Peak Limit (+{peak} Nm)", alpha=0.7)
            peak_line_neg, = ax.plot([], [], color="r", linestyle="--", linewidth=2, 
                                     label=f"Peak Limit (-{peak} Nm)", alpha=0.7)
            self.peak_lines[group_name] = [peak_line_pos, peak_line_neg]
            
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Torque (Nm)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"{group_name}", fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to ensure window is created
    
    def _update_realtime_plot(self) -> None:
        """Update real-time torque plots with latest data."""
        if len(self.torque_history) == 0:
            return
        
        torque_array = np.array(self.torque_history)
        time_array = np.array(self.time_history)
        
        axes_flat = self.axes.flatten()
        
        for idx, (group_name, group_info) in enumerate(self.joint_groups.items()):
            ax = axes_flat[idx]
            
            # Update each joint line
            for line_idx, (joint_idx, joint_name) in enumerate(zip(group_info["indices"], group_info["names"])):
                torque_data = torque_array[:, joint_idx]
                self.lines[group_name][line_idx].set_data(time_array, torque_data)
            
            # Update peak limit lines
            peak = group_info["peak"]
            x_range = [time_array[0], time_array[-1]] if len(time_array) > 0 else [0, 1]
            self.peak_lines[group_name][0].set_data(x_range, [peak, peak])
            self.peak_lines[group_name][1].set_data(x_range, [-peak, -peak])
            
            # Update axis limits
            if len(time_array) > 0:
                ax.set_xlim([time_array[0], time_array[-1]])
                # Auto-scale y-axis based on current data
                joint_torques = torque_array[:, group_info["indices"]]
                y_min = np.min(joint_torques) - 10
                y_max = np.max(joint_torques) + 10
                # Ensure peak limits are visible
                y_min = min(y_min, -peak - 10)
                y_max = max(y_max, peak + 10)
                ax.set_ylim([y_min, y_max])
            
            # Update title with max torque info
            if len(torque_array) > 0:
                max_torque = np.max(np.abs(torque_array[:, group_info["indices"]]))
                peak = group_info["peak"]
                if max_torque > peak:
                    ax.set_title(f"{group_name} (⚠️ MAX: {max_torque:.2f} Nm > {peak} Nm)", 
                               fontsize=12, fontweight="bold", color="red")
                else:
                    ax.set_title(f"{group_name} (MAX: {max_torque:.2f} Nm)", 
                               fontsize=12, fontweight="bold", color="green")
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update
    
    def _plot_torque_curves(self) -> None:
        """Plot torque curves for all joints, grouped by hip, knee, and ankle."""
        if len(self.torque_history) == 0:
            print("[WARNING] No torque data recorded for plotting.")
            return
        
        # Convert to numpy arrays
        torque_array = np.array(self.torque_history)  # Shape: (num_steps, 12)
        time_array = np.array(self.time_history)  # Shape: (num_steps,)
        
        # Define peak torque limits (Nm) - typical values for humanoid robots
        # These can be adjusted based on actual motor specifications
        # Peak torque limits from XML file (actuatorfrcrange values)
        peak_limits = {
            "hip_pitch": 330.0,  # Nm (from XML: actuatorfrcrange="-330 330")
            "hip_roll": 330.0,   # Nm (from XML: actuatorfrcrange="-330 330")
            "hip_yaw": 115.0,    # Nm (from XML: actuatorfrcrange="-115 115")
            "knee": 330.0,       # Nm (from XML: actuatorfrcrange="-330 330")
            "ankle_pitch": 75.0, # Nm (from XML: actuatorfrcrange="-75 75")
            "ankle_roll": 75.0,  # Nm (from XML: actuatorfrcrange="-75 75")
        }
        
        # Group joints by type
        joint_groups = {
            "Hip Pitch": {
                "indices": [0, 1],  # left_hip_pitch, right_hip_pitch
                "names": ["Left Hip Pitch", "Right Hip Pitch"],
                "peak": peak_limits["hip_pitch"],
            },
            "Hip Roll": {
                "indices": [2, 3],  # left_hip_roll, right_hip_roll
                "names": ["Left Hip Roll", "Right Hip Roll"],
                "peak": peak_limits["hip_roll"],
            },
            "Hip Yaw": {
                "indices": [4, 5],  # left_hip_yaw, right_hip_yaw
                "names": ["Left Hip Yaw", "Right Hip Yaw"],
                "peak": peak_limits["hip_yaw"],
            },
            "Knee": {
                "indices": [6, 7],  # left_knee, right_knee
                "names": ["Left Knee", "Right Knee"],
                "peak": peak_limits["knee"],
            },
            "Ankle Pitch": {
                "indices": [8, 9],  # left_ankle_pitch, right_ankle_pitch
                "names": ["Left Ankle Pitch", "Right Ankle Pitch"],
                "peak": peak_limits["ankle_pitch"],
            },
            "Ankle Roll": {
                "indices": [10, 11],  # left_ankle_roll, right_ankle_roll
                "names": ["Left Ankle Roll", "Right Ankle Roll"],
                "peak": peak_limits["ankle_roll"],
            },
        }
        
        # Create subplots for each joint group
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle("Joint Torque Curves (Time vs Torque)", fontsize=16, fontweight="bold")
        
        axes_flat = axes.flatten()
        
        for idx, (group_name, group_info) in enumerate(joint_groups.items()):
            ax = axes_flat[idx]
            
            # Plot each joint in the group
            for joint_idx, joint_name in zip(group_info["indices"], group_info["names"]):
                torque_data = torque_array[:, joint_idx]
                ax.plot(time_array, torque_data, label=joint_name, linewidth=1.5, alpha=0.8)
            
            # Add peak limit lines
            peak = group_info["peak"]
            ax.axhline(y=peak, color="r", linestyle="--", linewidth=2, label=f"Peak Limit (+{peak} Nm)", alpha=0.7)
            ax.axhline(y=-peak, color="r", linestyle="--", linewidth=2, label=f"Peak Limit (-{peak} Nm)", alpha=0.7)
            
            # Check for violations
            max_torque = np.max(np.abs(torque_array[:, group_info["indices"]]))
            if max_torque > peak:
                ax.set_title(f"{group_name} (⚠️ MAX: {max_torque:.2f} Nm > {peak} Nm)", 
                           fontsize=12, fontweight="bold", color="red")
            else:
                ax.set_title(f"{group_name} (MAX: {max_torque:.2f} Nm)", 
                           fontsize=12, fontweight="bold", color="green")
            
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Torque (Nm)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim([time_array[0], time_array[-1]])
        
        plt.tight_layout()
        
        # Save figure
        output_path = "torque_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Torque curves saved to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("TORQUE SUMMARY STATISTICS")
        print("="*80)
        for group_name, group_info in joint_groups.items():
            joint_torques = torque_array[:, group_info["indices"]]
            max_abs_torque = np.max(np.abs(joint_torques))
            peak = group_info["peak"]
            violation = max_abs_torque > peak
            status = "⚠️ EXCEEDED" if violation else "✓ OK"
            print(f"{group_name:15s}: Max = {max_abs_torque:7.2f} Nm, Peak Limit = {peak:6.1f} Nm  {status}")
        print("="*80)
        
        # Show plot (non-blocking)
        plt.show(block=False)

    def _setup_keyboard_listener(self) -> None:
        step = float(self.cfg.commands.step)

        def on_press(key):
            try:
                if key.char == "8":
                    self._adjust_command_vel(0, step)
                elif key.char == "2":
                    self._adjust_command_vel(0, -step)
                elif key.char == "4":
                    self._adjust_command_vel(1, -step)
                elif key.char == "6":
                    self._adjust_command_vel(1, step)
                elif key.char == "7":
                    self._adjust_command_vel(2, -step)
                elif key.char == "9":
                    self._adjust_command_vel(2, step)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


class _TorchScriptPolicy:
    def __init__(self, policy_path: str):
        self._module = torch.jit.load(policy_path, map_location="cpu")
        self._module.eval()

    def __call__(self, obs_history: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.tensor(obs_history, dtype=torch.float32)
            try:
                out = self._module(obs)
            except Exception:
                out = self._module(obs.unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.detach().cpu().numpy()


def _load_policy(policy_path: str):
    suffix = os.path.splitext(policy_path)[1].lower()

    if suffix in {".pt", ".pth", ".jit"} or zipfile.is_zipfile(policy_path):
        return _TorchScriptPolicy(policy_path)

    raise RuntimeError(
        f"Unknown policy format for '{policy_path}'. Expected a TorchScript file (.pt, .pth, .jit)."
    )


def _parse_args() -> argparse.Namespace:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run robot1_6 sim2sim (TorchScript policy + MuJoCo MJCF).")
    parser.add_argument(
        "--task",
        type=str,
        default="robot1_6_walk",
        choices=["robot1_6_walk", "robot1_6_run"],
        help="Task type: 'robot1_6_walk' or 'robot1_6_run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(root_dir, "legged_lab/assets/robot1_6/mjcf/Robot1_6.xml"),
        help="Path to MuJoCo MJCF model XML.",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=0,
        choices=[0, 1],
        help="Fixed action delay in control steps (training randomizes in [0,1]).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.policy is None:
        # Default to robot1_6_walk policy
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        args.policy = os.path.join(root_dir, "Exported_policy", "robot1_6_walk.pt")

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    cfg = SimToSimCfg()
    cfg.sim.sim_duration = float(args.duration)
    cfg.sim.action_delay_steps = int(args.action_delay_steps)

    # Set gait parameters according to task
    if args.task == "robot1_6_walk":
        cfg.robot.gait_air_ratio_l = 0.38
        cfg.robot.gait_air_ratio_r = 0.38
        cfg.robot.gait_phase_offset_l = 0.38
        cfg.robot.gait_phase_offset_r = 0.88
        cfg.robot.gait_cycle = 0.80
    elif args.task == "robot1_6_run":
        cfg.robot.gait_air_ratio_l = 0.6
        cfg.robot.gait_air_ratio_r = 0.6
        cfg.robot.gait_phase_offset_l = 0.6
        cfg.robot.gait_phase_offset_r = 0.1
        cfg.robot.gait_cycle = 0.4

    print(f"[INFO] Task preset: {args.task.upper()}")
    print(f"[INFO] Policy: {args.policy}")
    print(f"[INFO] Model: {args.model}")
    print(
        "[INFO] Keys: 8/2 inc/dec vx, 6/4 inc/dec vy, 9/7 inc/dec yaw_rate (NumPad digits, or regular digits)."
    )

    MujocoRunner(cfg=cfg, policy_path=args.policy, model_path=args.model).run()
