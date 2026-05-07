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

"""Configuration for Robot1_6 (lower body only, 12 DOF).

The following configurations are available:

* :obj:`ROBOT1_6_CFG`: Robot1_6 lower body robot with 12 joints

Reference: Custom robot design
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

ROBOT1_6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/robot1_6/usd/Robot1_6.usd",
        # fix_base=False,
        # replace_cylinders_with_capsules=False,
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
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            "left_hip_pitch_joint": -0.20,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": -0.05,
            "left_knee_joint": 0.32,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.20,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.05,
            "right_knee_joint": -0.32,
            "right_ankle_pitch_joint": 0.15,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0}, 
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            # Joint limits from URDF: effort=330, velocity=12 (for all leg joints)
            # Torque limits reduced to 70% of URDF values
            effort_limit_sim={
                ".*_hip_pitch_joint": 231,  # 330 * 0.7
                ".*_hip_roll_joint": 231,   # 330 * 0.7
                ".*_hip_yaw_joint": 231,    # 330 * 0.7
                ".*_knee_joint": 231,       # 330 * 0.7
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 8.4,   # 12.0 * 0.7
                ".*_hip_roll_joint": 8.4,    # 12.0 * 0.7
                ".*_hip_yaw_joint": 8.4,     # 12.0 * 0.7
                ".*_knee_joint": 8.4,        # 12.0 * 0.7
            },
            stiffness={
                ".*_hip_pitch_joint": 70,  # Match sim2sim.py kp=40 80 40 60 70 100 250 300
                ".*_hip_roll_joint": 140,   # Match sim2sim.py kp=90 180 99 140 140 200 300 350
                ".*_hip_yaw_joint": 45,    # Match sim2sim.py kp=40 40 40 40 45 60 60 60
                ".*_knee_joint": 140,       # Match sim2sim.py kp=90 180 99 140 140 200 200 250
            },
            damping={
                ".*_hip_pitch_joint": 5.0,   # Match sim2sim.py kd=4.0 8.0 2.55 5.0 20 25
                ".*_hip_roll_joint": 8.0,  # Match sim2sim.py kd=6.5 12.5 6.3 9.4 8.0 25 30
                ".*_hip_yaw_joint": 3.0,     # Match sim2sim.py kd=4.0 4.0 2.55 3.0 3.0 3.0
                ".*_knee_joint": 8.0     # Match sim2sim.py kd=6.5 12.5 6.3 9.4 8.0 10 15
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            # Joint limits from URDF: effort=75, velocity=12 (for all ankle joints)
            # Torque limits reduced to 70% of URDF values
            effort_limit_sim={
                ".*_ankle_pitch_joint": 52.5,  # 75 * 0.7
                ".*_ankle_roll_joint": 52.5,   # 75 * 0.7
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 8.4,   # 12.0 * 0.7
                ".*_ankle_roll_joint": 8.4,    # 12.0 * 0.7
            },
            stiffness={
                ".*_ankle_pitch_joint": 34,   # Match sim2sim.py kp=20 40 28 34 50 70
                ".*_ankle_roll_joint": 24,    # Match sim2sim.py kp=20 20 28 24 30 50
            },
            damping={
                ".*_ankle_pitch_joint": 2.4,  # Match sim2sim.py kd=1.5 3.0 1.8 2.4 3.0
                ".*_ankle_roll_joint": 1.65,   # Match sim2sim.py kd=1.5 1.5 1.8 1.65 1.5
            },
        ),
    },
)