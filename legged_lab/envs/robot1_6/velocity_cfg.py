# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# Pure RL Locomotion configuration for Robot1_6 robot.
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
from legged_lab.assets.robot1_6 import ROBOT1_6_CFG
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
class Robot1_6VelocityRewardCfg:
    """
    Pure RL Velocity Tracking reward configuration for Robot1_6.
    Reference: Q1VelocityRewardCfg
    """
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活,避免站立时惩罚
    )
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    )
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.5, "command_threshold": 0.1},
    # )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "command_threshold": 0.1},
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_world_exp,
    #     weight=1.0,
    #     params={"std": 0.5, "command_threshold": 0.1},  # ✨仅在运动时激活
    # )
    is_alive = RewTerm(func=mdp.is_alive, weight=0.15)

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
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                # joint_names=[".*hip_roll_joint", ".*hip_yaw_joint"],
                joint_names=[".*hip_yaw_joint"],
            ),
        },
    )
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
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.98},  # Robot1_6 standing height
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
        weight=-0.2,
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
    
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6, params={"delta_t": 0.02})

# @configclass
# class Robot1_6VelocityEnvCfg:
#     """Pure RL Velocity Tracking environment configuration for Robot1_6."""

#     device: str = "cuda:0"

#     scene: BaseSceneCfg = BaseSceneCfg(
#         max_episode_length_s=20.0,
#         num_envs=4096,
#         env_spacing=2.5,
#         robot=ROBOT1_6_CFG,
#         terrain_type="plane",
#         terrain_generator=None,
#         max_init_terrain_level=0,
#         height_scanner=HeightScannerCfg(
#             enable_height_scan=False,
#             prim_body_name="base_link",
#             resolution=0.1,
#             size=(1.6, 1.0),
#             debug_vis=False,
#             drift_range=(0.0, 0.0),
#         ),
#     )

#     robot: RobotCfg = RobotCfg(
#         actor_obs_history_length=5,  # 5 frames as per Q1 reference
#         critic_obs_history_length=5,
#         action_scale=0.25,
#         terminate_contacts_body_names=[
#             ".*_knee_link",
#             "base_link",
#         ],
#         feet_body_names=[".*_ankle_roll_link"],
#         # min_base_height=0.2,
#         # max_tilt=0.8,
#     )

#     reward = Robot1_6VelocityRewardCfg()

#     normalization: NormalizationCfg = NormalizationCfg(
#         obs_scales=ObsScalesCfg(
#             lin_vel=1.0,
#             ang_vel=0.2,  # Scaled as per Q1 reference
#             projected_gravity=1.0,
#             commands=1.0,
#             joint_pos=1.0,
#             joint_vel=0.05,  # Scaled as per Q1 reference
#             actions=1.0,
#             height_scan=1.0,
#         ),
#         clip_observations=100.0,
#         clip_actions=100.0,
#         height_scan_offset=0.5,
#     )

#     commands: CommandsCfg = CommandsCfg(
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.02,  # Reduced standing ratio for locomotion
#         rel_heading_envs=1.0,
#         heading_command=False,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=CommandRangesCfg(
#             lin_vel_x=(-0.3, 0.3),
#             lin_vel_y=(-0.2, 0.2),
#             ang_vel_z=(-0.2, 0.2),
#             heading=(-math.pi, math.pi),
#         ),
#         limit_ranges=CommandRangesCfg(
#             lin_vel_x=(-0.5, 1.0),
#             lin_vel_y=(-0.3, 0.3),
#             ang_vel_z=(-0.2, 0.2),
#             heading=(-math.pi, math.pi),
#         ),
#         # Ignore tiny linear-velocity commands
#         lin_vel_threshold=0.2,
#         curriculum_cmd_threshold=0.25,
#         curriculum_success_threshold=0.8,
#         curriculum_delta=0.1,
#         curriculum=True,
#     )

#     noise: NoiseCfg = NoiseCfg(
#         add_noise=True,
#         noise_scales=NoiseScalesCfg(
#             lin_vel=0.2,
#             ang_vel=0.2,
#             projected_gravity=0.05,
#             joint_pos=0.01,
#             joint_vel=1.5,
#             height_scan=0.1,
#         ),
#     )

#     domain_rand: DomainRandCfg = DomainRandCfg(
#         events=EventCfg(
#             physics_material=EventTerm(
#                 func=mdp.randomize_rigid_body_material,
#                 mode="startup",
#                 params={
#                     "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#                     "static_friction_range": (0.3, 1.6),
#                     "dynamic_friction_range": (0.3, 1.2),
#                     "restitution_range": (0.0, 0.5),
#                     "num_buckets": 64,
#                 },
#             ),
#             add_joint_default_pos=EventTerm(
#                 func=mdp.randomize_joint_default_pos,
#                 mode="startup",
#                 params={
#                     "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
#                     "pos_distribution_params": (-0.01, 0.01),
#                     "operation": "add",
#                 },
#             ),
#             base_com=EventTerm(
#                 func=mdp.randomize_rigid_body_com,
#                 mode="startup",
#                 params={
#                     "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#                     "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
#                 },
#             ),
#             add_base_mass=None,
#             reset_base=None,
#             reset_robot_joints=None,
#             push_robot=EventTerm(
#                 func=mdp.push_by_setting_velocity,
#                 mode="interval",
#                 interval_range_s=(1.0, 3.0),
#                 params={"velocity_range": VELOCITY_RANGE},
#             ),
#         ),
#         action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
#     )

#     sim: SimCfg = SimCfg(
#         dt=0.005,
#         decimation=4,
#         physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
#     )


# @configclass
# class Robot1_6VelocityAgentCfg(RslRlOnPolicyRunnerCfg):
#     """Pure RL PPO Agent configuration for Robot1_6 velocity tracking."""

#     seed = 42
#     device = "cuda:0"
#     num_steps_per_env = 24
#     max_iterations = 50000
#     save_interval = 100
#     empirical_normalization = False

#     policy = RslRlPpoActorCriticCfg(
#         class_name="ActorCritic",
#         init_noise_std=1.0,
#         noise_std_type="scalar",
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )

#     algorithm = RslRlPpoAlgorithmCfg(
#         class_name="PPO",  # Standard PPO, not AMPPPO
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,  # Higher entropy for more exploration
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#         normalize_advantage_per_mini_batch=False,
#         symmetry_cfg=None,
#         rnd_cfg=None,
#     )

#     clip_actions = None

#     # Standard OnPolicyRunner, not AmpOnPolicyRunner
#     runner_class_name = "OnPolicyRunner"
#     experiment_name = "robot1_6_velocity"
#     run_name = ""
#     logger = "tensorboard"
#     neptune_project = "robot1_6_velocity"
#     wandb_project = "robot1_6_velocity"

#     resume = False
#     load_run = ".*"
#     load_checkpoint = "model_.*.pt"
