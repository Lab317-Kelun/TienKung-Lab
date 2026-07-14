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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def _get_command(env, command_name: str | None = None) -> torch.Tensor:
    """
    获取command的辅助函数,兼容多种环境接口。
    
    Args:
        env: 环境实例
        command_name: 命令名称(可选)
        
    Returns:
        命令张量 [num_envs, 3] (通常为 [vx, vy, vyaw])
    """
    if command_name is not None and hasattr(env, "command_manager"):
        return env.command_manager.get_command(command_name)
    if hasattr(env, "command_generator"):
        return env.command_generator.command
    if hasattr(env, "command_manager"):
        return env.command_manager.get_command(command_name)
    raise AttributeError("Environment does not provide command_manager or command_generator.")


def _upright_gate(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """
    计算直立门控系数,当机器人倒下时减少奖励。
    
    原理:
        - projected_gravity_b[:, 2] 是重力在body frame z轴的投影
        - 当机器人直立时: projected_gravity_z ≈ -1.0
        - 当机器人倒下时: projected_gravity_z ≈ 0.0
    
    Returns:
        门控系数 [0, 1],直立时为1.0,倒下时为0.0
    """
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    command_threshold: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = _get_command(env, command_name)
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    if command_threshold is not None:
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    return reward * _upright_gate(env)


def track_lin_vel_x_yaw_frame_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    command_threshold: float | None = None,
) -> torch.Tensor:
    """Reward tracking of linear velocity command (x axis) in the gravity aligned robot frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = _get_command(env, command_name)
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error_x = torch.square(command[:, 0] - vel_yaw[:, 0])
    reward = torch.exp(-lin_vel_error_x / std**2)
    if command_threshold is not None:
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    return reward * _upright_gate(env)


def track_lin_vel_y_yaw_frame_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    command_threshold: float | None = None,
) -> torch.Tensor:
    """Reward tracking of linear velocity command (y axis) in the gravity aligned robot frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = _get_command(env, command_name)
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error_y = torch.square(command[:, 1] - vel_yaw[:, 1])
    reward = torch.exp(-lin_vel_error_y / std**2)
    if command_threshold is not None:
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    return reward * _upright_gate(env)


def track_ang_vel_z_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = _get_command(env, command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if command_threshold is not None:
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    return reward * _upright_gate(env)

def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    command_threshold: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = _get_command(env, command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if command_threshold is not None:
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    return reward * _upright_gate(env)


def track_heading_exp(
    env: BaseEnv | TienKungEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float | None = None,
) -> torch.Tensor:
    """Reward tracking of target heading (world yaw) using exponential kernel.

    Requires ``heading_command=True`` on the velocity command generator so that
    ``heading_target`` is resampled each command interval.
    """
    if not hasattr(env, "command_generator"):
        raise AttributeError("track_heading_exp requires env.command_generator.")

    cmd_gen = env.command_generator
    if not cmd_gen.cfg.heading_command:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    heading_error = math_utils.wrap_to_pi(cmd_gen.heading_target - asset.data.heading_w)
    reward = torch.exp(-torch.square(heading_error) / std**2)

    reward = reward * cmd_gen.is_heading_env.float()
    reward = reward * (~cmd_gen.is_standing_env).float()

    if command_threshold is not None:
        command = cmd_gen.command
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()

    return reward * _upright_gate(env)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize joint deviation; when command_threshold is set, only penalize near-zero commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if command_threshold is None:
        return torch.sum(torch.abs(angle), dim=1)
    command = _get_command(env)
    zero_flag = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < command_threshold
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[1], env.right_leg_ids[1]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def feet_distance_lateral(env: TienKungEnv, min_distance: float = 0.2, max_distance: float = 0.35) -> torch.Tensor:
    """Penalize feet crossing by maintaining lateral distance between feet.
    
    This reward prevents feet from crossing during turning maneuvers by penalizing
    when the lateral (y-direction) distance between feet is too small or too large.
    
    Parameters
    ----------
    env : TienKungEnv
        The environment instance.
    min_distance : float
        Minimum acceptable lateral distance between feet. Default is 0.2m.
    max_distance : float
        Maximum acceptable lateral distance between feet. Default is 0.35m.
    
    Returns
    -------
    torch.Tensor
        Penalty reward (negative values) when feet are too close or too far apart.
    """
    # Get foot positions in world frame
    left_foot_pos_w = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :]
    right_foot_pos_w = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :]
    
    # Translate to root frame
    left_foot_pos_rel = left_foot_pos_w - env.robot.data.root_link_pos_w[:, :]
    right_foot_pos_rel = right_foot_pos_w - env.robot.data.root_link_pos_w[:, :]
    
    # Transform to body frame
    root_quat = env.robot.data.root_link_quat_w[:, :]
    left_foot_pos_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), left_foot_pos_rel)
    right_foot_pos_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), right_foot_pos_rel)
    
    # Calculate lateral (y-direction) distance
    lateral_distance = torch.abs(left_foot_pos_b[:, 1] - right_foot_pos_b[:, 1])
    
    # Penalize if too close (feet crossing) or too far apart
    penalty = (
        torch.clamp(lateral_distance - min_distance, max=0.0) +
        torch.clamp(-lateral_distance + max_distance, max=0.0)
    )
    
    return penalty


def knee_distance_lateral(env: TienKungEnv, min_distance: float = 0.2, max_distance: float = 0.35) -> torch.Tensor:
    """Penalize knee crossing by maintaining lateral distance between knees.
    
    This reward prevents knees from crossing during turning maneuvers by penalizing
    when the lateral (y-direction) distance between knees is too small or too large.
    
    Parameters
    ----------
    env : TienKungEnv
        The environment instance.
    min_distance : float
        Minimum acceptable lateral distance between knees. Default is 0.2m.
    max_distance : float
        Maximum acceptable lateral distance between knees. Default is 0.35m.
    
    Returns
    -------
    torch.Tensor
        Penalty reward (negative values) when knees are too close or too far apart.
    """
    # Find knee body IDs
    knee_body_ids, _ = env.robot.find_bodies(
        name_keys=["left_knee_link", "right_knee_link"],
        preserve_order=True,
    )
    
    # Get knee positions in world frame
    left_knee_pos_w = env.robot.data.body_pos_w[:, knee_body_ids[0], :]
    right_knee_pos_w = env.robot.data.body_pos_w[:, knee_body_ids[1], :]
    
    # Translate to root frame
    left_knee_pos_rel = left_knee_pos_w - env.robot.data.root_link_pos_w[:, :]
    right_knee_pos_rel = right_knee_pos_w - env.robot.data.root_link_pos_w[:, :]
    
    # Transform to body frame
    root_quat = env.robot.data.root_link_quat_w[:, :]
    left_knee_pos_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), left_knee_pos_rel)
    right_knee_pos_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), right_knee_pos_rel)
    
    # Calculate lateral (y-direction) distance
    lateral_distance = torch.abs(left_knee_pos_b[:, 1] - right_knee_pos_b[:, 1])
    
    # Penalize if too close (knees crossing) or too far apart
    penalty = (
        torch.clamp(lateral_distance - min_distance, max=0.0) +
        torch.clamp(-lateral_distance + max_distance, max=0.0)
    )
    
    return penalty


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """
    惩罚步态摆动相时的脚部力。
    
    在摆动相（脚部在空中）时，脚部不应该有接触力。奖励摆动相时脚部力接近零。
    当命令速度接近零时，屏蔽此奖励（避免在静止时鼓励抬腿）。
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位和脚部力）
    """
    # 速度掩码：当速度接近零时，屏蔽步态奖励
    nonzero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    gait_reward = left_frc_score + right_frc_score
    # 仅在非零速度时应用步态奖励
    return gait_reward * nonzero_flag


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """
    惩罚步态支撑相时的脚部速度。
    
    在支撑相（脚部接触地面）时，脚部应该保持相对静止。奖励支撑相时脚部速度接近零。
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位和脚部速度）
    """
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    gait_reward = left_spd_score + right_spd_score
    return gait_reward 


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """
    奖励步态支撑相时的适当支撑力。
    
    在支撑相（脚部接触地面）时，脚部应该有足够的支撑力。奖励支撑相时脚部有适当的接触力。
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位和脚部力）
    """
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    gait_reward = left_frc_score + right_frc_score
    return gait_reward 

def stand_still(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    惩罚在零命令时的运动（通过接触力判断）。
    
    当命令速度很小时，惩罚那些没有接触地面或接触力很小的脚，
    鼓励所有脚都接触地面，保持稳定。
    
    通用性：✅ 适用于所有机器人（通过 sensor_cfg.body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取脚部的接触力 Z 分量（垂直方向）
    contact_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # 计算接触力小于0.1的脚的数量（即没有接触或接触力很小的脚）
    contacts = torch.sum(contact_forces_z < 0.1, dim=-1)
    # 当命令速度很小时（< 0.1），返回这个数量作为惩罚
    zero_flag = torch.norm(env.command_generator.command[:, :3], dim=1) < 0.1
    return contacts * zero_flag


def stand_still_lin_vel_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚在零命令时的身体线速度。
    
    当命令速度很小时，惩罚身体在xy平面的线速度，鼓励机器人保持静止。
    
    通用性：✅ 适用于所有机器人
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算身体在xy平面的线速度（在yaw frame中）
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_xy = torch.sum(torch.square(vel_yaw[:, :2]), dim=1)
    # 当命令速度很小时（< 0.1），惩罚线速度
    zero_flag = torch.norm(env.command_generator.command[:, :3], dim=1) < 0.1
    return lin_vel_xy * zero_flag


def stand_still_ang_vel_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚在零命令时的身体角速度。
    
    当命令速度很小时，惩罚身体的角速度，鼓励机器人保持静止。
    
    通用性：✅ 适用于所有机器人
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算身体的角速度
    ang_vel = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    # 当命令速度很小时（< 0.1），惩罚角速度
    zero_flag = torch.norm(env.command_generator.command[:, :3], dim=1) < 0.1
    return ang_vel * zero_flag


def stand_still_joint_vel_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚在零命令时的关节速度。
    
    当命令速度很小时，惩罚关节速度，鼓励机器人保持静止姿态。
    
    通用性：✅ 适用于所有机器人
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节速度的L2范数
    joint_vel = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    # 当命令速度很小时（< 0.1），惩罚关节速度
    zero_flag = torch.norm(env.command_generator.command[:, :3], dim=1) < 0.1
    return joint_vel * zero_flag


def stand_still_joint_pos_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚在零命令时的关节位置偏离默认位置。
    
    当命令速度很小时，惩罚关节位置偏离默认位置，鼓励机器人保持静止时的标准姿态。
    
    通用性：✅ 适用于所有机器人
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 计算关节位置偏离默认位置的L2范数
    joint_pos_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_l2 = torch.sum(torch.square(joint_pos_error), dim=1)
    # 当命令速度很小时（< 0.1），惩罚关节位置偏离
    zero_flag = torch.norm(env.command_generator.command[:, :3], dim=1) < 0.1
    return joint_pos_l2 * zero_flag


def feet_clearance(
    env: TienKungEnv,
    target_feet_height: float = 0.03,
    height_tolerance: float = 0.01,
    contact_threshold: float = 5.0,
    delta_t: float = 0.02,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Reward for maintaining proper foot clearance during swing phase.
    
    This implementation maintains cumulative foot height (like Gym version):
    - Uses cumulative foot height (feet_height += delta_z)
    - Resets height on contact (feet_height *= ~contact)
    - Uses world coordinate z-position minus 0.05
    - Uses gait_clock to get swing mask
    
    Parameters
    ----------
    env : TienKungEnv
        The environment instance.
    target_feet_height : float
        Target height for feet during swing phase (in meters). Default is 0.06m (matching Gym).
    height_tolerance : float
        Tolerance around target height for reward (in meters). Default is 0.01m.
    contact_threshold : float
        Contact force threshold to determine if foot is in contact (in Newtons). Default is 5.0N.
    delta_t : float
        Transition width for gait phase boundaries. Default is 0.02.
    sensor_cfg : SceneEntityCfg
        Configuration for contact sensor to detect foot contact. Default uses ankle roll links.
    
    Returns
    -------
    torch.Tensor
        Reward value (positive when feet are at target height during swing phase).
    """
    # Compute feet contact mask (same as Gym: contact_forces[:, feet_indices, 2] > 5.0)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get contact forces for feet (shape: [num_envs, num_feet, 3])
    foot_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # Check z-component of contact force (same as Gym)
    contact = foot_contact_forces[:, :, 2] > contact_threshold
    
    # Get the z-position of the feet in world frame and subtract 0.05 (same as Gym)
    # Gym: feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
    feet_z_w = env.robot.data.body_pos_w[:, env.feet_body_ids, 2] - 0.05
    
    # Compute the change in z-position (same as Gym)
    # Gym: delta_z = feet_z - self.last_feet_z
    delta_z = feet_z_w - env.last_feet_z
    
    # Update cumulative foot height (same as Gym)
    # Gym: self.feet_height += delta_z
    env.feet_height += delta_z
    
    # Update last_feet_z (same as Gym)
    # Gym: self.last_feet_z = feet_z
    env.last_feet_z = feet_z_w
    
    # Get swing phase masks using gait_clock (not using 1 - _get_gait_phase())
    left_swing_mask, _ = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)
    right_swing_mask, _ = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)
    swing_mask = torch.stack([left_swing_mask, right_swing_mask], dim=1)  # [num_envs, 2]
    
    # Feet height should be close to target feet height at the peak (same as Gym)
    # Gym: rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
    rew_pos = torch.abs(env.feet_height - target_feet_height) < height_tolerance
    
    # Multiply by swing mask and sum over feet (same as Gym)
    # Gym: rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    
    # Disable reward when standing still (command velocity is near zero)
    # 当命令速度很小时（静止状态），不给予脚部离地高度奖励
    zero_vel_flag = torch.norm(env.command_generator.command[:, :3], dim=1) >= 0.1
    rew_pos = rew_pos * zero_vel_flag
    
    # Reset feet_height on contact (same as Gym)
    # Gym: self.feet_height *= ~contact
    env.feet_height *= ~contact
    
    return rew_pos


def base_height_l2(
    env: BaseEnv | TienKungEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward for maintaining base_link at target height.
    Penalizes deviation from target height using squared error.
    
    Args:
        env: The environment.
        target_height: Target height for the base_link (in meters).
        asset_cfg: Scene entity configuration for the robot.
        
    Returns:
        Squared height error for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get base_link height (z position in world frame)
    base_height = asset.data.root_pos_w[:, 2]
    # Calculate squared error from target height
    height_error = torch.square(base_height - target_height)
    return height_error


# =====================================================
# Pure RL Locomotion Reward Functions (from Q1)
# Reference: unitree_rl_lab / RoboCup_Lab
# =====================================================


def feet_gait(
    env: BaseEnv | TienKungEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
    command_threshold: float | None = None,
) -> torch.Tensor:
    """
    基于步态周期的接触状态奖励(纯RL方法)。
    奖励脚部接触状态与期望的步态相位匹配。
    
    原理:
        - 根据episode时间计算当前步态相位 [0, 1]
        - 判断该相位下脚应该处于支撑相(stance)还是摆动相(swing)
        - 对比实际接触状态,匹配则给予奖励
    
    优势:
        - 无需预定义复杂的摆动/支撑相掩码
        - 基于时间的简单周期信号
        - 可以与gait_feet_frc_perio等函数配合使用
    
    Args:
        env: 环境实例
        period: 步态周期(秒),例如0.8s
        offset: 每条腿的相位偏移 [0.0, 0.5] 用于交替步态
        sensor_cfg: 接触传感器配置
        threshold: 支撑相占比,例如0.55表示55%时间在地面
        command_name: 命令名称(可选)
        command_threshold: 仅在命令足够大时激活奖励(可选)
    
    Returns:
        奖励张量 [num_envs]
        
    Example:
        ```python
        # 交替步态,周期0.8s,支撑相55%
        feet_gait = RewTerm(
            func=mdp.feet_gait,
            weight=1.0,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],  # 左右脚相位差180度
                "threshold": 0.55,
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
                "command_threshold": 0.1,
            }
        )
        ```
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    
    # 计算全局相位 [0, 1]
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    
    # 为每条腿计算相位(加上offset)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)
    
    # 计算奖励
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        # 期望状态: phase < threshold 时应该接触地面
        is_stance_expected = leg_phase[:, i] < threshold
        # 实际状态
        is_contact_actual = is_contact[:, i]
        # 匹配奖励: 使用XOR取反,匹配时为True
        reward += ~(is_stance_expected ^ is_contact_actual)
    
    # 可选: 仅在命令足够大时激活
    if command_threshold is not None:
        command = _get_command(env, command_name)
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    
    return reward


def foot_clearance_reward(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    command_threshold: float | None = None,
    velocity_threshold: float | None = None,
) -> torch.Tensor:
    """
    摆动相脚部离地高度奖励(高级版本,来自Q1)。
    鼓励脚部在移动时达到目标离地高度。
    
    特点:
        - 基于脚部速度的连续激活(而非二元掩码)
        - 可选的地形高度补偿(支持复杂地形)
        - 多重过滤机制(命令阈值、速度阈值)
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置,包含脚部body names
        target_height: 目标脚部高度(米),例如0.1m
        std: 指数核的标准差
        tanh_mult: 速度门控的tanh乘数
        sensor_cfg: 可选的地形传感器,用于高度补偿
        command_name: 可选的命令名称
        command_threshold: 最小命令大小以激活奖励
        velocity_threshold: 最小基座速度以激活奖励
    
    Returns:
        奖励张量 [num_envs]
        
    Example:
        ```python
        foot_clearance_reward = RewTerm(
            func=mdp.foot_clearance_reward,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "target_height": 0.1,
                "std": 0.05,
                "tanh_mult": 3.0,
                "command_threshold": 0.1,
            }
        )
        ```
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # 可选: 地形高度补偿(支持复杂地形)
    if sensor_cfg is not None:
        sensor = env.scene.sensors[sensor_cfg.name]
        terrain_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
        foot_z = foot_z - terrain_z
    
    # 计算高度误差
    foot_z_target_error = torch.square(foot_z - target_height)
    
    # 基于脚部速度的连续门控(速度快时权重高)
    foot_velocity = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    foot_velocity_tanh = torch.tanh(tanh_mult * foot_velocity)
    
    # 综合奖励
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    
    # 可选: 命令阈值过滤
    if command_threshold is not None:
        command = _get_command(env, command_name)
        cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (cmd_norm > command_threshold).float()
    
    # 可选: 实际速度阈值过滤
    if velocity_threshold is not None:
        vel_norm = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) + torch.abs(asset.data.root_ang_vel_b[:, 2])
        reward = reward * (vel_norm > velocity_threshold).float()
    
    return reward


def joint_vel_l2(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    惩罚关节速度的L2范数。
    用于平滑关节运动,减少震颤。
    
    Args:
        env: 环境实例
        asset_cfg: 机器人配置
    
    Returns:
        惩罚张量 [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def is_alive(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """
    生存奖励: 每个时间步给予常数奖励。
    鼓励机器人保持存活,延长episode长度。
    
    Args:
        env: 环境实例
    
    Returns:
        全1奖励张量 [num_envs]
    """
    return torch.ones(env.num_envs, device=env.device)