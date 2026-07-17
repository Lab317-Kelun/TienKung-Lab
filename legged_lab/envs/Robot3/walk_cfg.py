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

import math

from legged_lab.envs.Robot3.velocity_cfg import Robot3VelocityRewardCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.Robot3 import ROBOT3_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    ActionNoiseCfg,
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85 #0.85


@configclass
class LiteRewardCfg:
    # ========== 速度跟踪奖励 (增强版: 添加command_threshold和自动直立门控) ==========
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=2.0,
        params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活,避免站立时惩罚
    )
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_yaw_frame_exp,
        weight=2.0,
        params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 从 -1.0 提高到 -2.0，增强垂直速度惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # 从 -0.05 提高到 -0.1，增强翻滚/俯仰惩罚
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*_knee_link", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )  # 从 -2.0 提高到 -3.0，增强基座姿态稳定性
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)  # 从 -1.0 提高到 -1.5，增强身体垂直度
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 600,  # 根据机器人质量 51.6kg 调整：单脚支撑 ~506N，运动峰值 ~600-800N
            "max_reward": 500,  # 相应提高最大惩罚值
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    # Q1注释掉了feet_stumble(与feet_slide功能重复)
    # feet_stumble = RewTerm(
    #     func=mdp.feet_stumble,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    # )
    # Q1注释掉了dof_pos_limits(有joint_deviation就够了)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # ========== 关节偏离惩罚 (增强版: 添加command_threshold,仅在低速时惩罚) ==========
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                ],
            ),
            "command_threshold": 0.15,  # ✨仅在低速(cmd<0.15)时惩罚偏离
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            ),
            "command_threshold": 0.15,  # ✨仅在低速时惩罚偏离
        },
    )
    # ========== 步态奖励 (参考Q1,保留原有周期奖励+新增纯RL步态) ==========
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})
    # ✨新增: 纯RL步态奖励(来自Q1,基于时间周期,配合使用)
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,  # 与GaitCfg.gait_cycle一致
            "offset": [0.0, 0.5],  # 交替步态
            "threshold": 0.55,  # 支撑相55%
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "command_threshold": 0.1,
        },
    )

    # ========== 特定关节约束 (参考Q1保留) ==========
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)
    feet_distance_lateral = RewTerm(
        func=mdp.feet_distance_lateral,
        weight=5.0,  # 增加权重以更严格惩罚双脚并拢
        params={"min_distance": 0.25, "max_distance": 0.35}, 
    )
    knee_distance_lateral = RewTerm(
        func=mdp.knee_distance_lateral,
        weight=5.0,
        params={"min_distance": 0.18, "max_distance": 0.25},  
    )
    # 静止时的接触奖励（惩罚在零命令时没有接触地面的脚）
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
        },
    )
    # ========== 脚部离地高度奖励 (Q1没有此项,但建议保留) ==========
    # Q1只用步态奖励隐式控制,但显式的脚部离地奖励可以帮助训练
    # 注释掉旧版本,只保留新的高级版本
    # feet_clearance = RewTerm(
    #     func=mdp.feet_clearance,
    #     weight=1.0,
    #     params={
    #         "target_feet_height": 0.03,
    #         "height_tolerance": 0.01,
    #         "contact_threshold": 5.0,
    #         "delta_t": 0.02,
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #     },
    # )
    
    # ✨保留新增的高级版本(基于速度的连续激活,更平滑)
    foot_clearance_reward = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,  # 替代旧版本,权重提高
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.03,
            "std": 0.01,
            "tanh_mult": 3.0,
            "command_threshold": 0.1,
        },
    )
    # base_link 高度奖励（保持机器人在目标高度 0.96m）
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.90},
    )
    
    # ========== 新增辅助奖励 (来自Q1) ==========
    # 关节速度平滑(减少震颤)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 生存奖励(鼓励延长episode)
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.1,
    )


@configclass
class Robot3WalkFlatEnvCfg:
    amp_motion_files_display = ["legged_lab/envs/Robot3/datasets/motion_visualization/AI2Robot_Walk.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ROBOT3_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=[
            "pelvis",
            ".*_knee_link",
            "waist_yaw_link",
            ".*_shoulder_roll_link",
            ".*_elbow_link",
        ],
        feet_body_names=[".*_ankle_roll_link"],
    )
    reward = LiteRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 0.8), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
        velocity_threshold=0.0,  # 速度命令阈值：当x、y、yaw方向的绝对值小于此值时，设置为0
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=0.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.5, 1.25),
                    "dynamic_friction_range": (0.5, 1.25),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 64,
                },
            ),
            scale_link_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-1.0, 3.0),
                    "operation": "add",
                },
            ),
            base_com=EventTerm(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.8, 1.2),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 10.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
            robot_joint_stiffness_and_damping=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.9, 1.1),
                    "damping_distribution_params": (0.9, 1.1),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            ),
        ),
        action_delay=ActionDelayCfg(enable=True, params={"max_delay": 1, "min_delay": 0}),
        action_noise=ActionNoiseCfg(enable=False, noise_scale=0.02),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Robot3WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 1000
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "robot3_walk"
    run_name = ""
    logger = "wandb"
    neptune_project = "robot3_walk"
    wandb_project = "robot3_walk"  # 项目名称
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/Robot3/datasets/motion_amp_expert/AI2Robot_Walk.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 12  # legs only (waist locked)
    reduce_amp_reward_at_zero_velocity = False  # 当命令速度接近零时，是否减少AMP奖励
  

@configclass
class Robot3WalkAmpFlatEnvCfg(Robot3WalkFlatEnvCfg):
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=[
            "pelvis",
            ".*_knee_link",
            "waist_yaw_link",
            ".*_shoulder_roll_link",
            ".*_elbow_link",
        ],
        feet_body_names=[".*_ankle_roll_link"],
    )
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    reward = Robot3VelocityRewardCfg()


@configclass
class Robot3WalkAmpAgentCfg(Robot3WalkAgentCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "robot3_walk_amp"
