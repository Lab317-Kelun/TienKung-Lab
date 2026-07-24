# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Configuration for Robot3 humanoid (13-DOF lower-body policy; upper body PD-locked)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

ROBOT3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_ASSET_DIR}/Robot3/urdf/Robot3.urdf",
        usd_dir=f"{ISAAC_ASSET_DIR}/Robot3/usd",
        fix_base=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            # Required by UrdfFileCfg validation. Runtime PD is set by ImplicitActuatorCfg below.
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
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
            "left_shoulder_pitch_joint": -0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": -0.0,
            "right_shoulder_roll_joint": -0.0,
            "right_shoulder_yaw_joint": -0.0,
            "right_elbow_joint": -0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_roll_joint": 0.0,
            "head_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Motor model parameters (armature/friction) are taken from X6-16
        # identification reports. Torque/velocity limits are scaled by 0.9.
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim={".*_hip_pitch_joint": 405},  # 450 * 0.9
            velocity_limit_sim={".*_hip_pitch_joint": 6.786},  # 7.54 * 0.9
            stiffness={".*_hip_pitch_joint": 700.0},
            damping={".*_hip_pitch_joint": 10.0},
            # X15: J=0.51 kg·m², Fc=2.311 Nm, Fv=0.6365 Nm/(rad/s)
            armature=0.51,
            friction=2.311,
        ),
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 288,  # 320 * 0.9
                ".*_knee_joint": 288,      # 320 * 0.9
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 11.781,  # 13.09 * 0.9
                ".*_knee_joint": 11.781,      # 13.09 * 0.9
            },
            stiffness={
                ".*_hip_roll_joint": 700.0,
                ".*_knee_joint": 700.0,
            },
            damping={
                ".*_hip_roll_joint": 10.0,
                ".*_knee_joint": 10.0,
            },
            # X12: J≈0.49 kg·m², Fc=2.827 Nm, Fv≈0
            armature=0.49,
            friction=2.827,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim={".*_hip_yaw_joint": 108},  # 120 * 0.9
            velocity_limit_sim={".*_hip_yaw_joint": 14.895},  # 16.55 * 0.9
            stiffness={".*_hip_yaw_joint": 500.0},
            damping={".*_hip_yaw_joint": 5.0},
            # X8: (armature from summary J≈0.0568), Fc=0.9233 Nm, Fv=0.1184 Nm/(rad/s)
            armature=0.0568,
            friction=0.9233,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 52.5 * 1.713,
                ".*_ankle_roll_joint": 52.5* 1.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 8.4 * 0.660,
                ".*_ankle_roll_joint": 8.4 * 1.167,
            },
            stiffness={
                ".*_ankle_pitch_joint": 80,   #50 70 90 110 130 150  80 * 4.589
                ".*_ankle_roll_joint": 80,     #50 70 90 110 130 150 80 * 1.467
            },
            damping={
                ".*_ankle_pitch_joint": 5,   #3.125 4.375 5.625 6.875 8.125 9.375 5 * 4.589
                ".*_ankle_roll_joint": 5,   #3.125 4.375 5.625 6.875 8.125 9.375  5 * 1.467
            },
            # X6: J=0.0245 kg·m², Fc=0.5666 Nm, Fv=0.06872 Nm/(rad/s)
            armature={
                ".*_ankle_pitch_joint": 0.003 *4.589,
                ".*_ankle_roll_joint": 0.003 * 1.467,
            },
            friction={
                ".*_ankle_pitch_joint": 0.5666,
                ".*_ankle_roll_joint": 0.5666,
            },
        ),
        # Only assign waist motor params for now; other upper-body joints keep None.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim={"waist_yaw_joint": 180.0},  # 200 * 0.9
            velocity_limit_sim={"waist_yaw_joint": 7.542},  # 8.38 * 0.9
            stiffness=500.0,
            damping=5.0,
            armature=None,
            friction=None,
        ),
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
                "head_.*_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 186.3,  # 207 * 0.9
                ".*_shoulder_roll_joint": 186.3,   # 207 * 0.9
                ".*_shoulder_yaw_joint": 91.8,     # 102 * 0.9
                ".*_elbow_joint": 4500.0,          # 5000 * 0.9
                ".*_wrist_roll_joint": 59.4,       # 66 * 0.9
                ".*_wrist_pitch_joint": 540.0,     # 600 * 0.9
                ".*_wrist_yaw_joint": 540.0,       # 600 * 0.9
                "head_.*_joint": 4.32,             # 4.8 * 0.9
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 2.826,  # 3.14 * 0.9
                ".*_shoulder_roll_joint": 2.826,   # 3.14 * 0.9
                ".*_shoulder_yaw_joint": 3.771,    # 4.19 * 0.9
                ".*_elbow_joint": 3.6,             # 4.0 * 0.9
                ".*_wrist_roll_joint": 3.861,      # 4.29 * 0.9
                ".*_wrist_pitch_joint": 3.15,      # 3.5 * 0.9
                ".*_wrist_yaw_joint": 3.15,        # 3.5 * 0.9
                "head_.*_joint": 5.562,            # 6.18 * 0.9
            },
            stiffness={
                ".*_shoulder_pitch_joint": 120.0,   #80.0->120.0
                ".*_shoulder_roll_joint": 60.0,   #40.0->60.0
                ".*_shoulder_yaw_joint": 45.0,   #30.0->45.0
                ".*_elbow_joint": 40.0,          #30.0->40.0
                ".*_wrist_.*_joint": 20.0,       #20.0->20.0
                "head_.*_joint": 20.0,           #20.0->20.0
            },
            damping={
                ".*_shoulder_pitch_joint": 4.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_.*_joint": 1.0,
                "head_.*_joint": 2.0,
            },
            armature=None,
            friction=None,
        ),
    },
)
