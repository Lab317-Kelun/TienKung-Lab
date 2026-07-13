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
    from legged_lab.envs.robot1_6.robot1_6_env import Robot1_6Env


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | Robot1_6Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跟踪水平面（XY）线性速度（在偏航坐标系中）。
    
    奖励机器人按照命令的水平速度移动。使用指数衰减函数，速度误差越小奖励越高。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | Robot1_6Env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    跟踪绕Z轴的角速度（在世界坐标系中）。
    
    奖励机器人按照命令的角速度旋转。使用指数衰减函数，角速度误差越小奖励越高。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚垂直方向（Z轴）的线性速度。
    
    鼓励机器人保持水平移动，避免不必要的上下运动。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚绕X和Y轴的角速度（俯仰和翻滚）。
    
    鼓励机器人保持直立，避免前后左右倾斜。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚能量消耗（力矩 × 关节速度）。
    
    鼓励机器人使用更少的能量，提高运动效率。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚关节加速度的L2范数。
    
    鼓励平滑的运动，减少突然的加速度变化。
    
    通用性：✅ 适用于所有机器人（通过 asset_cfg.joint_ids 配置，不依赖固定顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | Robot1_6Env) -> torch.Tensor:
    """
    惩罚动作变化率（相邻两步动作的差异）。
    
    鼓励平滑的控制，减少动作的剧烈变化。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | Robot1_6Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚不期望的接触（如膝盖、躯干等部位接触地面）。
    
    鼓励只有脚部接触地面，避免其他部位碰撞。
    
    通用性：✅ 适用于所有机器人（通过 sensor_cfg.body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | Robot1_6Env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚飞行状态（没有脚部接触地面）。
    
    鼓励机器人至少有一只脚接触地面，避免完全悬空。
    
    通用性：✅ 适用于所有机器人（通过 sensor_cfg.body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚非水平姿态（重力在机器人坐标系中的XY分量）。
    
    鼓励机器人保持水平，避免前后左右倾斜。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | Robot1_6Env) -> torch.Tensor:
    """
    惩罚非超时终止的episode（如摔倒、碰撞等）。
    
    当episode因为失败而终止（而非正常超时）时给予惩罚。
    
    通用性：✅ 适用于所有机器人（不依赖关节顺序）
    """
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | Robot1_6Env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    奖励单腿支撑时的空中时间或接触时间。
    
    在单腿支撑阶段，奖励较长的空中时间（摆动腿）或接触时间（支撑腿），鼓励稳定的步态。
    仅在命令速度非零时给予奖励。
    
    通用性：✅ 适用于所有双足机器人（通过 body_ids 配置，不依赖关节顺序）
    """
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
    env: BaseEnv | Robot1_6Env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚脚部滑动（接触地面时的水平速度）。
    
    鼓励脚部在接触地面时保持稳定，减少滑动。
    
    通用性：✅ 适用于所有机器人（通过 body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | Robot1_6Env, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """
    惩罚过大的脚部接触力（超过阈值后）。
    
    鼓励适度的接触力，避免过大的冲击力。
    
    通用性：✅ 适用于所有机器人（通过 body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚关节偏离默认位置（仅在命令速度接近零时）。
    
    当机器人静止时，鼓励关节回到默认位置。
    
    通用性：✅ 适用于所有机器人（通过 joint_ids 配置，不依赖固定顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def stand_still_joint_vel_l2(env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    惩罚关节速度（仅在命令速度接近零时）。
    
    当机器人静止时，鼓励所有关节速度接近零，保持稳定。
    
    通用性：✅ 适用于所有机器人（通过 joint_ids 配置，不依赖固定顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.square(joint_vel), dim=1) * zero_flag


def stand_still(
    env: BaseEnv | Robot1_6Env, sensor_cfg: SceneEntityCfg
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


def body_orientation_l2(
    env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚身体姿态的非水平分量（使用指定身体部位的姿态）。
    
    鼓励指定身体部位（如躯干）保持水平。
    
    通用性：✅ 适用于所有机器人（通过 body_ids 配置，不依赖关节顺序）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | Robot1_6Env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚脚部打滑/绊倒（水平力远大于垂直力）。
    
    检测脚部是否受到过大的水平力，表明可能发生打滑或绊倒。
    
    通用性：✅ 适用于所有机器人（通过 body_ids 配置，不依赖关节顺序）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | Robot1_6Env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """
    惩罚双脚距离过近（小于阈值时）。
    
    鼓励保持合理的步宽，避免双脚过于靠近导致不稳定。
    
    通用性：✅ 适用于所有双足机器人（通过 body_ids 配置，不依赖关节顺序）
    """
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: Robot1_6Env) -> torch.Tensor:
    """
    惩罚踝关节的力矩。
    
    鼓励减少踝关节的力矩输出，提高运动效率。
    
    通用性：✅ 适用于所有机器人（通过 ankle_joint_ids 查找，不依赖固定索引）
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: Robot1_6Env) -> torch.Tensor:
    """
    惩罚踝关节的动作幅度。
    
    鼓励减少踝关节的动作，提高运动稳定性。
    
    通用性：✅ 适用于所有机器人（通过 ankle_joint_ids 查找，不依赖固定索引）
    """
    action_indices = getattr(env, "action_ankle_indices", None)
    if action_indices is not None:
        return torch.sum(torch.abs(env.action[:, action_indices]), dim=1)
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: Robot1_6Env) -> torch.Tensor:
    """
    惩罚髋关节侧摆（roll）的动作幅度。
    
    鼓励减少髋关节侧摆动作，提高运动稳定性。
    
    通用性：✅ 适用于 Robot1_6（使用 left_leg_ids[1] 和 right_leg_ids[1]，GMR 顺序）
    注意：Robot1_6 关节顺序为 hip_pitch[0], hip_roll[1], hip_yaw[2], knee[3], ankle_pitch[4], ankle_roll[5]
    """
    action_indices = getattr(env, "action_hip_roll_indices", None)
    if action_indices is not None:
        return torch.sum(torch.abs(env.action[:, action_indices]), dim=1)
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[1], env.right_leg_ids[1]]]), dim=1)


def hip_yaw_action(env: Robot1_6Env) -> torch.Tensor:
    """
    惩罚髋关节偏航（yaw）的动作幅度。
    
    鼓励减少髋关节偏航动作，提高运动稳定性。
    
    通用性：✅ 适用于 Robot1_6（使用 left_leg_ids[2] 和 right_leg_ids[2]，GMR 顺序）
    注意：Robot1_6 关节顺序为 hip_pitch[0], hip_roll[1], hip_yaw[2], knee[3], ankle_pitch[4], ankle_roll[5]
    """
    action_indices = getattr(env, "action_hip_yaw_indices", None)
    if action_indices is not None:
        return torch.sum(torch.abs(env.action[:, action_indices]), dim=1)
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: Robot1_6Env) -> torch.Tensor:
    """
    惩罚脚部Y方向距离偏差（当命令的Y速度较小时）。
    
    在侧向移动较小时，鼓励保持合理的步宽（默认0.299米）。
    
    通用性：✅ 适用于所有双足机器人（通过 feet_body_ids 查找，不依赖关节顺序）
    注意：0.299 是硬编码的期望步宽，可能需要根据机器人调整
    """
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    生成周期性步态时钟信号，用于脚部摆动相和支撑相。
    
    此函数构造两个相位相关的信号：
    - `I_frc`: 在摆动相激活（用于惩罚地面力）
    - `I_spd`: 在支撑相激活（用于惩罚脚部速度）
    
    摆动相和支撑相之间的过渡在 `delta_t` 范围内平滑插值，以创建可微分的过渡。
    
    参数
    ----------
    phase : torch.Tensor
        归一化的步态相位 [0, 1]，形状: [num_envs]
    air_ratio : torch.Tensor
        步态周期中摆动相的比例，形状: [num_envs]
    delta_t : float
        相位边界周围的过渡宽度，用于平滑插值
    
    返回
    -------
    I_frc : torch.Tensor
        基于步态的摆动相时钟信号，范围 [0, 1]，形状: [num_envs]
    I_spd : torch.Tensor
        基于步态的支撑相时钟信号，范围 [0, 1]，形状: [num_envs]
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位）
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


def gait_feet_frc_perio(env: Robot1_6Env, delta_t: float = 0.02) -> torch.Tensor:
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


def gait_feet_spd_perio(env: Robot1_6Env, delta_t: float = 0.02) -> torch.Tensor:
    """
    惩罚步态支撑相时的脚部速度。
    
    在支撑相（脚部接触地面）时，脚部应该保持相对静止。奖励支撑相时脚部速度接近零。
    当命令速度接近零时，屏蔽此奖励（避免在静止时鼓励周期性运动）。
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位和脚部速度）
    """
    # 速度掩码：当速度接近零时，屏蔽步态奖励
    nonzero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    gait_reward = left_spd_score + right_spd_score
    # 仅在非零速度时应用步态奖励
    return gait_reward * nonzero_flag


def gait_feet_frc_support_perio(env: Robot1_6Env, delta_t: float = 0.02) -> torch.Tensor:
    """
    奖励步态支撑相时的适当支撑力。
    
    在支撑相（脚部接触地面）时，脚部应该有足够的支撑力。奖励支撑相时脚部有适当的接触力。
    当命令速度接近零时，屏蔽此奖励（避免在静止时鼓励周期性运动）。
    
    通用性：✅ 适用于所有双足机器人（不依赖关节顺序，只依赖步态相位和脚部力）
    """
    # 速度掩码：当速度接近零时，屏蔽步态奖励
    nonzero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    gait_reward = left_frc_score + right_frc_score
    # 仅在非零速度时应用步态奖励
    return gait_reward * nonzero_flag
