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
import re
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard


# Isaac Lab policy joint order (robot3_env.py): right_leg(6) + left_leg(6).
POLICY_JOINT_NAMES = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

# Waist + upper body are PD-locked at default pose during training.
LOCKED_JOINT_NAMES = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "head_yaw_joint",
    "head_roll_joint",
    "head_pitch_joint",
]

ALL_ACTUATED_JOINT_NAMES = POLICY_JOINT_NAMES + LOCKED_JOINT_NAMES

# Default joint positions from legged_lab/assets/Robot3/robot3.py::ROBOT3_CFG.init_state.joint_pos
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.25,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.07,
    "left_knee_joint": 0.5,
    "left_ankle_pitch_joint": -0.25,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.25,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.07,
    "right_knee_joint": 0.5,
    "right_ankle_pitch_joint": -0.25,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_roll_joint": 0.0,
    "head_pitch_joint": 0.0,
}

# PD gains from legged_lab/assets/Robot3/robot3.py::ROBOT3_CFG.actuators
POLICY_JOINT_PD = {
    ".*_hip_pitch_joint": (700.0, 10.0),
    ".*_hip_roll_joint": (700.0, 10.0),
    ".*_knee_joint": (700.0, 10.0),
    ".*_hip_yaw_joint": (500.0, 5.0),
    ".*_ankle_pitch_joint": (30.0, 2.5),
    ".*_ankle_roll_joint": (16.8, 1.4),
}
LOCKED_JOINT_PD = {
    "waist_yaw_joint": (500.0, 5.0),
    ".*_shoulder_pitch_joint": (120.0, 4.0),
    ".*_shoulder_roll_joint": (60.0, 3.0),
    ".*_shoulder_yaw_joint": (45.0, 2.0),
    ".*_elbow_joint": (40.0, 2.0),
    ".*_wrist_roll_joint": (20.0, 1.0),
    ".*_wrist_pitch_joint": (20.0, 1.0),
    ".*_wrist_yaw_joint": (20.0, 1.0),
    "head_.*_joint": (20.0, 2.0),
}
PD_GAINS_BY_JOINT = {**POLICY_JOINT_PD, **LOCKED_JOINT_PD}


def _lookup_pd_gain(joint_name: str) -> tuple[float, float]:
    for pattern, gains in PD_GAINS_BY_JOINT.items():
        if re.fullmatch(pattern, joint_name):
            return gains
    raise KeyError(f"No PD gains configured for joint '{joint_name}'.")


def _quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q_xyzw[-1]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def _base_quat_xyzw(qpos: np.ndarray) -> np.ndarray:
    quat_wxyz = qpos[3:7].astype(np.float64)
    return quat_wxyz[[1, 2, 3, 0]]


class SimToSimCfg:
    """Sim2Sim configuration for Robot3.

    Must be kept consistent with:
    - `legged_lab/envs/Robot3/robot3_env.py::compute_current_observations`
    - `legged_lab/envs/Robot3/walk_cfg.py::Robot3WalkAmpFlatEnvCfg`
    - `legged_lab/assets/Robot3/robot3.py::ROBOT3_CFG`
    """

    class sim:
        sim_duration = 100.0
        dt = 0.005
        decimation = 4

        num_action = 12
        num_obs_per_step = 45
        actor_obs_history_length = 10

        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        action_delay_steps = 0

        obs_scales = {
            "ang_vel": 1.0,
            "projected_gravity": 1.0,
            "commands": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "actions": 1.0,
        }

    class robot:
        init_base_height = 1.1

    class commands:
        lin_vel_x_range = (-0.6, 0.8)
        lin_vel_y_range = (-0.5, 0.5)
        ang_vel_z_range = (-0.5, 0.5)
        step = 0.1


class MujocoRunner:
    """Sim2Sim runner for Robot3 TorchScript policy + MuJoCo MJCF."""

    def __init__(self, cfg: SimToSimCfg, policy_path: str, model_path: str):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = float(self.cfg.sim.dt)

        self.policy = _load_policy(policy_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False

        self._init_mappings_and_indices()
        self._init_gains_and_limits()
        self._init_buffers()
        self._init_robot_state()
        self._validate_policy_io()

    def _init_mappings_and_indices(self) -> None:
        self.mujoco_actuator_names: list[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)
        ]

        if self.model.nu != len(ALL_ACTUATED_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(ALL_ACTUATED_JOINT_NAMES)} actuators, got {self.model.nu} in MJCF."
            )

        missing = [name for name in ALL_ACTUATED_JOINT_NAMES if name not in self.mujoco_actuator_names]
        if missing:
            raise ValueError(f"MuJoCo model missing actuators for joints: {missing}")

        for sensor_name in ("imu_angvel", "imu_linvel"):
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) < 0:
                raise ValueError(f"MuJoCo model missing required sensor: {sensor_name}")

        mujoco_name_to_idx = {name: idx for idx, name in enumerate(self.mujoco_actuator_names)}
        self.mujoco_to_isaac_idx = np.array(
            [mujoco_name_to_idx[name] for name in POLICY_JOINT_NAMES], dtype=np.int32
        )
        self.locked_mujoco_idx = np.array(
            [mujoco_name_to_idx[name] for name in LOCKED_JOINT_NAMES], dtype=np.int32
        )

        self._qpos_idx = np.zeros(self.model.nu, dtype=np.int32)
        self._qvel_idx = np.zeros(self.model.nu, dtype=np.int32)
        for act_id, joint_name in enumerate(self.mujoco_actuator_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Actuator '{joint_name}' does not map to a joint with the same name.")
            self._qpos_idx[act_id] = int(self.model.jnt_qposadr[joint_id])
            self._qvel_idx[act_id] = int(self.model.jnt_dofadr[joint_id])

        self.default_dof_pos_mujoco = np.array(
            [DEFAULT_JOINT_POS[name] for name in self.mujoco_actuator_names], dtype=np.float64
        )
        self.default_dof_pos_isaac = np.array(
            [DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_NAMES], dtype=np.float64
        )

    def _init_gains_and_limits(self) -> None:
        kp_list = []
        kd_list = []
        for act_id, name in enumerate(self.mujoco_actuator_names):
            kp, kd_pd = _lookup_pd_gain(name)
            # Fv is modeled as passive joint damping in MJCF (system ID).
            # Subtract it from external PD kd to avoid double-counting viscous damping.
            passive_fv = float(self.model.dof_damping[self._qvel_idx[act_id]])
            kp_list.append(kp)
            kd_list.append(max(0.0, kd_pd - passive_fv))
        self.kp = np.array(kp_list, dtype=np.float64)
        self.kd = np.array(kd_list, dtype=np.float64)
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

    def _init_buffers(self) -> None:
        self.dt = float(self.cfg.sim.dt)
        self.control_dt = float(self.cfg.sim.decimation * self.cfg.sim.dt)

        self.dof_pos_mujoco = np.zeros(self.model.nu, dtype=np.float64)
        self.dof_vel_mujoco = np.zeros(self.model.nu, dtype=np.float64)

        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float64)
        self._action_cmd_buf = [np.zeros(self.cfg.sim.num_action, dtype=np.float64) for _ in range(2)]
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

        self.torque_history: list[np.ndarray] = []
        self.time_history: list[float] = []

    def _init_robot_state(self) -> None:
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_base_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._qpos_idx] = self.default_dof_pos_mujoco
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _validate_policy_io(self) -> None:
        obs_dim = self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length
        dummy_obs = np.zeros(obs_dim, dtype=np.float32)
        action = np.asarray(self.policy(dummy_obs), dtype=np.float64).reshape(-1)
        if action.shape[0] < self.cfg.sim.num_action:
            raise ValueError(
                f"Policy output dim {action.shape[0]} < expected action dim {self.cfg.sim.num_action}."
            )
        print(f"[INFO] Policy IO check passed: obs={obs_dim}, action={action.shape[0]}")

    def _read_actuated_joint_state_mujoco_order(self) -> tuple[np.ndarray, np.ndarray]:
        self.dof_pos_mujoco[:] = self.data.qpos[self._qpos_idx]
        self.dof_vel_mujoco[:] = self.data.qvel[self._qvel_idx]
        return self.dof_pos_mujoco, self.dof_vel_mujoco

    def _get_base_ang_vel_body(self) -> np.ndarray:
        imu_angvel_world = self.data.sensor("imu_angvel").data.astype(np.float64)
        return _quat_rotate_inverse(_base_quat_xyzw(self.data.qpos), imu_angvel_world)

    def _get_base_lin_vel_body(self) -> np.ndarray:
        imu_linvel_world = self.data.sensor("imu_linvel").data.astype(np.float64)
        return _quat_rotate_inverse(_base_quat_xyzw(self.data.qpos), imu_linvel_world)

    def get_obs(self) -> np.ndarray:
        ang_vel = self._get_base_ang_vel_body()
        projected_gravity = _quat_rotate_inverse(
            _base_quat_xyzw(self.data.qpos), np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )

        dof_pos_mujoco, dof_vel_mujoco = self._read_actuated_joint_state_mujoco_order()
        dof_pos_isaac = dof_pos_mujoco[self.mujoco_to_isaac_idx] - self.default_dof_pos_isaac
        dof_vel_isaac = dof_vel_mujoco[self.mujoco_to_isaac_idx]

        obs = np.concatenate(
            [
                ang_vel * self.cfg.sim.obs_scales["ang_vel"],
                projected_gravity * self.cfg.sim.obs_scales["projected_gravity"],
                self.command_vel * self.cfg.sim.obs_scales["commands"],
                dof_pos_isaac * self.cfg.sim.obs_scales["joint_pos"],
                dof_vel_isaac * self.cfg.sim.obs_scales["joint_vel"],
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                * self.cfg.sim.obs_scales["actions"],
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        if obs.shape[0] != self.cfg.sim.num_obs_per_step:
            raise RuntimeError(
                f"Observation dim mismatch: built {obs.shape[0]}, expected {self.cfg.sim.num_obs_per_step}."
            )

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs
        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def _compute_target_pos_mujoco_order(self, action_isaac: np.ndarray) -> np.ndarray:
        target_pos_mujoco = self.default_dof_pos_mujoco.copy()
        target_pos_isaac = action_isaac * self.cfg.sim.action_scale + self.default_dof_pos_isaac
        target_pos_mujoco[self.mujoco_to_isaac_idx] = target_pos_isaac
        return target_pos_mujoco

    def _position_pd_torque(self, target_pos_mujoco: np.ndarray) -> np.ndarray:
        dof_pos_mujoco, dof_vel_mujoco = self._read_actuated_joint_state_mujoco_order()
        torque = self.kp * (target_pos_mujoco - dof_pos_mujoco) - self.kd * dof_vel_mujoco
        return np.clip(torque, self.ctrl_lo, self.ctrl_hi)

    def run(self) -> None:
        self._setup_keyboard_listener()
        self.listener.start()

        print_counter = 0
        print_interval = 50

        while self.data.time < float(self.cfg.sim.sim_duration):
            obs_history = self.get_obs()

            action_np = np.asarray(self.policy(obs_history), dtype=np.float64).reshape(-1)
            action_cmd = np.clip(
                action_np[: self.cfg.sim.num_action], -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions
            )

            delay = int(self.cfg.sim.action_delay_steps)
            if delay not in (0, 1):
                raise ValueError(f"action_delay_steps must be 0 or 1, got {delay}.")
            self._action_cmd_buf.pop(0)
            self._action_cmd_buf.append(action_cmd)
            self.action = self._action_cmd_buf[0 if delay == 1 else 1].copy()

            target_pos_mujoco = self._compute_target_pos_mujoco_order(self.action)

            for _ in range(int(self.cfg.sim.decimation)):
                step_start_time = time.time()
                self.data.ctrl[:] = self._position_pd_torque(target_pos_mujoco)

                self.torque_history.append(self.data.ctrl[self.mujoco_to_isaac_idx].copy())
                self.time_history.append(self.data.time)

                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print_counter += 1
            if print_counter >= print_interval:
                print_counter = 0
                base_lin_vel_b = self._get_base_lin_vel_body()
                base_ang_vel_b = self._get_base_ang_vel_body()
                print(f"\n[Time: {self.data.time:.2f}s]")
                print(
                    f"命令速度(body): vx={self.command_vel[0]:+.3f}, vy={self.command_vel[1]:+.3f}, "
                    f"yaw={self.command_vel[2]:+.3f}"
                )
                print(
                    f"实际速度(body): vx={base_lin_vel_b[0]:+.3f}, vy={base_lin_vel_b[1]:+.3f}, "
                    f"yaw={base_ang_vel_b[2]:+.3f}"
                )
                print(f"Base 高度: z={self.data.qpos[2]:+.3f} m")

        self.listener.stop()
        self.viewer.close()
        self._plot_torque_curves()

    def _adjust_command_vel(self, idx: int, increment: float) -> None:
        self.command_vel[idx] += float(increment)
        if idx == 0:
            lo, hi = self.cfg.commands.lin_vel_x_range
        elif idx == 1:
            lo, hi = self.cfg.commands.lin_vel_y_range
        else:
            lo, hi = self.cfg.commands.ang_vel_z_range
        self.command_vel[idx] = float(np.clip(self.command_vel[idx], lo, hi))

    def _plot_torque_curves(self) -> None:
        if len(self.torque_history) == 0:
            print("[WARNING] No torque data recorded for plotting.")
            return

        torque_array = np.array(self.torque_history)
        time_array = np.array(self.time_history)

        joint_groups = {
            "Hip Pitch": {"indices": [0, 6], "names": ["Right Hip Pitch", "Left Hip Pitch"]},
            "Hip Roll": {"indices": [1, 7], "names": ["Right Hip Roll", "Left Hip Roll"]},
            "Hip Yaw": {"indices": [2, 8], "names": ["Right Hip Yaw", "Left Hip Yaw"]},
            "Knee": {"indices": [3, 9], "names": ["Right Knee", "Left Knee"]},
            "Ankle Pitch": {"indices": [4, 10], "names": ["Right Ankle Pitch", "Left Ankle Pitch"]},
            "Ankle Roll": {"indices": [5, 11], "names": ["Right Ankle Roll", "Left Ankle Roll"]},
        }

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle("Robot3 Policy Joint Torque Curves", fontsize=16, fontweight="bold")
        axes_flat = axes.flatten()

        for idx, (group_name, group_info) in enumerate(joint_groups.items()):
            ax = axes_flat[idx]
            for joint_idx, joint_name in zip(group_info["indices"], group_info["names"]):
                ax.plot(time_array, torque_array[:, joint_idx], label=joint_name, linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Torque (Nm)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(group_name)

        plt.tight_layout()
        output_path = "robot3_torque_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Torque curves saved to: {output_path}")
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
    parser = argparse.ArgumentParser(description="Run Robot3 sim2sim (TorchScript policy + MuJoCo MJCF).")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported TorchScript policy.pt.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(root_dir, "legged_lab/assets/Robot3/mjcf/Robot3.xml"),
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

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    cfg = SimToSimCfg()
    cfg.sim.sim_duration = float(args.duration)
    cfg.sim.action_delay_steps = int(args.action_delay_steps)

    print("[INFO] Robot3 sim2sim")
    print(f"[INFO] Policy: {args.policy}")
    print(f"[INFO] Model: {args.model}")
    print(
        f"[INFO] obs={cfg.sim.num_obs_per_step * cfg.sim.actor_obs_history_length} "
        f"({cfg.sim.num_obs_per_step}x{cfg.sim.actor_obs_history_length}), action={cfg.sim.num_action}"
    )
    print("[INFO] Policy joints: right_leg(6) + left_leg(6)")
    print(f"[INFO] Locked joints: {len(LOCKED_JOINT_NAMES)} (waist + upper body, dedicated PD hold)")
    print("[INFO] Keys: 8/2 vx, 6/4 vy, 9/7 yaw")

    MujocoRunner(cfg=cfg, policy_path=args.policy, model_path=args.model).run()
