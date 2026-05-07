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
from tkinter.constants import FALSE

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
from legged_lab.assets.robot1_6 import ROBOT1_6_CFG
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
    gait_air_ratio_l: float = 0.6
    gait_air_ratio_r: float = 0.6
    gait_phase_offset_l: float = 0.6
    gait_phase_offset_r: float = 0.1
    gait_cycle: float = 0.4


@configclass
class LiteRewardCfg:
    track_lin_vel_x_exp = RewTerm(func=mdp.track_lin_vel_x_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_lin_vel_y_exp = RewTerm(func=mdp.track_lin_vel_y_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 从 -1.0 提高到 -2.0，增强垂直速度惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)  # 从 -0.05 提高到 -0.1，增强翻滚/俯仰惩罚
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*_knee_link", "base_link"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")}, weight=-3.0
    )  # 从 -2.0 提高到 -3.0，增强基座姿态稳定性
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)  # 从 -1.0 提高到 -1.5，增强身体垂直度
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
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
            "threshold": 700,  # 根据机器人质量 51.6kg 调整：单脚支撑 ~506N，运动峰值 ~600-800N
            "max_reward": 600,  # 相应提高最大惩罚值
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            )
        },
    )
    # 静止时的接触奖励（惩罚在零命令时没有接触地面的脚）
    # stand_still = RewTerm(
    #     func=mdp.stand_still,
    #     weight=-0.15,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #     },
    # )

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)

    feet_distance_lateral = RewTerm(
        func=mdp.feet_distance_lateral,
        weight=1.5,  # 增加权重以更严格惩罚双脚并拢
        params={"min_distance": 0.28, "max_distance": 0.38},  # 根据初始距离0.3528m调整：min=0.28, max=0.38
    )
    knee_distance_lateral = RewTerm(
        func=mdp.knee_distance_lateral,
        weight=2.0,
        params={"min_distance": 0.33, "max_distance": 0.45},  # 根据初始距离0.4132m调整：min=0.33, max=0.45
    )

@configclass
class Robot1_6RunFlatEnvCfg:
    amp_motion_files_display = ["legged_lab/envs/robot1_6/datasets/motion_visualization/run.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ROBOT1_6_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="base_link",
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
        terminate_contacts_body_names=[".*_knee_link", "base_link"],
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
        heading_command=False,  # 直接使用角速度命令，不从朝向计算
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 2.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi) 
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=FALSE,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.5,
            projected_gravity=0.03,
            joint_pos=0.02,
            joint_vel=0.1,
            height_scan=0.1,
        ),
    )
    # domain_rand: DomainRandCfg = DomainRandCfg(
    #     events=EventCfg(
    #         physics_material=EventTerm(
    #             func=mdp.randomize_rigid_body_material,
    #             mode="startup",
    #             params={
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #                 "static_friction_range": (0.3, 1.2),
    #                 "dynamic_friction_range": (0.3, 1.2),
    #                 "restitution_range": (0.0, 0.005),
    #                 "num_buckets": 64,
    #             },
    #         ),
    #         add_base_mass=EventTerm(
    #             func=mdp.randomize_rigid_body_mass,
    #             mode="startup",
    #             params={
    #                 "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #                 "mass_distribution_params": (-5.0, 5.0),
    #                 "operation": "add",
    #             },
    #         ),
    #         reset_base=EventTerm(
    #             func=mdp.reset_root_state_uniform,
    #             mode="reset",
    #             params={
    #                 "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #                 "velocity_range": {
    #                     "x": (-0.3, 0.3),
    #                     "y": (-0.3, 0.3),
    #                     "z": (-0.0, 0.0),
    #                     "roll": (-0.0, 0.0),
    #                     "pitch": (-0.0, 0.0),
    #                     "yaw": (-0.3, 0.3),
    #                 },
    #             },
    #         ),
    #         reset_robot_joints=EventTerm(
    #             func=mdp.reset_joints_by_scale,
    #             mode="reset",
    #             params={
    #                 "position_range": (1.0, 1.0),
    #                 "velocity_range": (-0.0, 0.0),
    #             },
    #         ),
    #         push_robot=EventTerm(
    #             func=mdp.push_by_setting_velocity,
    #             mode="interval",
    #             interval_range_s=(5.0, 10.0),
    #             params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    #         ),
    #         # 质心位移随机化
    #         randomize_com_displacement=EventTerm(
    #             func=mdp.randomize_rigid_body_com,
    #             mode="startup",
    #             params={
    #                 "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #                 "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
    #             },
    #         ),
    #         # PD控制器参数随机化 (Kp/Kd)
    #         randomize_actuator_gains=EventTerm(
    #             func=mdp.randomize_actuator_gains,
    #             mode="startup",
    #             params={
    #                 "asset_cfg": SceneEntityCfg("robot"),
    #                 "stiffness_distribution_params": (0.9, 1.1),
    #                 "damping_distribution_params": (0.9, 1.1),
    #                 "operation": "scale",
    #                 "distribution": "uniform",
    #             },
    #         ),
    #         # 链接质量随机化（所有链接）
    #         randomize_link_mass=EventTerm(
    #             func=mdp.randomize_rigid_body_mass,
    #             mode="startup",
    #             params={
    #                 "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #                 "mass_distribution_params": (0.8, 1.2),
    #                 "operation": "scale",
    #                 "distribution": "uniform",
    #             },
    #         ),
    #     ),
    #     action_delay=ActionDelayCfg(enable=False, params={"max_delay": 2, "min_delay": 0}),
    #     action_noise=ActionNoiseCfg(enable=False, noise_scale=0.02),
    # )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Robot1_6RunAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "run"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "run"
    wandb_project = "run"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/robot1_6/datasets/motion_amp_expert/run.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 12  # robot1_6 has 12 joints
