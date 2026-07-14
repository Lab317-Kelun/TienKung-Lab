# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# Pure RL Locomotion configuration for Robot3 robot.
# Reference: Q1_rl_lab Q1VelocityRewardCfg
# Pure RL (no AMP) with feet_gait reward

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.Robot3 import ROBOT3_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class Robot3VelocityRewardCfg:
    """
    Pure RL Velocity Tracking reward configuration for Robot3.
    Reference: Q1VelocityRewardCfg
    """
    # track_lin_vel_x_exp = RewTerm(
    #     func=mdp.track_lin_vel_x_yaw_frame_exp,
    #     weight=1.0,
    #     params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活,避免站立时惩罚
    # )
    # track_lin_vel_y_exp = RewTerm(
    #     func=mdp.track_lin_vel_y_yaw_frame_exp,
    #     weight=1.0,
    #     params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    # )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5, "command_threshold": 0.1},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5, "command_threshold": 0.1},
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_world_exp,
    #     weight=1.0,
    #     params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    # )
    is_alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # === Base penalties ===
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # === Energy and smoothness ===
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # === Joint constraints ===
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             # joint_names=[".*hip_roll_joint", ".*hip_yaw_joint"],
    #             joint_names=[".*hip_yaw_joint"],
    #         ),
    #     },
    # )
    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*ankle_roll_joint"],
            ),
        },
    )
    
    joint_deviation_heads = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*head_.*_joint"],
            ),
        },
    )
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*_joint",
    #             ],
    #         ),
    #     },
    # )
        
    stand_still = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_threshold": 0.1,
        },
    )
    stand_still_feet = RewTerm(
        func=mdp.stand_still,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
        },
    )

    # === Posture ===
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )
        
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 1.05},  # Robot3 standing height
    )

    # === Gait and feet ===
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "command_threshold": 0.1,
            "velocity_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # === Contact penalties ===
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"),
            "threshold": 1.0,
        },
    )
    # feet_force = RewTerm(
    #     func=mdp.body_force,
    #     weight=-3e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "threshold": 500,
    #         "max_reward": 400,
    #     },
    # )
    feet_distance_lateral = RewTerm(
        func=mdp.feet_distance_lateral,
        weight=5.0,  # 增加权重以更严格惩罚双脚并拢
        params={"min_distance": 0.22, "max_distance": 0.35}, 
    )
    knee_distance_lateral = RewTerm(
        func=mdp.knee_distance_lateral,
        weight=5.0,
        params={"min_distance": 0.20, "max_distance": 0.30},  
    )
    
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    # gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    # gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

