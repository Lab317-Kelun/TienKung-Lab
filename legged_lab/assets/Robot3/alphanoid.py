import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agile.rl_env.mdp.actuators.actuators_cfg import DelayedImplicitActuatorCfg


MIN_DELAY_PHY_STEPS = 0
MAX_DELAY_PHY_STEPS = 4


# ASSET_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "resource", "Alphanoid_V1.2")
# )
# ALPHANOID_USD_PATH = os.path.join(ASSET_DIR, "alphanoid2.usd")

# ASSET_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "resource", "Robot3")
# )
# ALPHANOID_USD_PATH = os.path.join(ASSET_DIR, "Alphanoid0623.usd")

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "resource", "Robot3.1")
)
# ALPHANOID_USD_PATH = os.path.join(ASSET_DIR, "usd", "Robot1_6.usd")
ALPHANOID_USD_PATH = os.path.join(ASSET_DIR, "Robot3.usd")

LEG_JOINT_NAMES = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
ALL_JOINT_NAMES = [
    ".*",
]
WAIST_JOINT_NAMES = [
    "waist.*",
]
ARM_JOINT_NAMES = [
    ".*_shoulder.*_joint",
    ".*_elbow_joint",
    ".*_wrist.*_joint",
]
HAND_JOINT_NAMES = [
]
HEAD_JOINT_NAMES = [
    "head.*_joint",
]
FEET_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
KNEE_LINK_NAMES = [
    "left_knee_link",
    "right_knee_link",
]
HAND_END_EFFECTOR_LINK_NAMES = [
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]
BASE_LINK_NAME = "pelvis"
DEFAULT_BASE_HEIGHT = 1.10
STAND_BASE_HEIGHT = 1.06
SQUAT_BASE_HEIGHT = 1.0
ALPHANOID_ACTION_SCALE_LOWER = {
    ".*_hip_pitch_joint": 0.25,
    ".*_hip_roll_joint": 0.20,
    ".*_hip_yaw_joint": 0.15,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": 0.20,
    ".*_ankle_roll_joint": 0.12,
}
UPPER_BODY_RANDOM_JOINT_LIMITS = {
    "waist_yaw_joint": (-2.6005406, 2.6005406),
    "head_yaw_joint": (-2.7925268, 2.7925268),
    "head_roll_joint": (-0.5235988, 0.5235988),
    "head_pitch_joint": (-0.5235988, 0.5235988),
    "left_shoulder_pitch_joint": (-3.1066861, 3.1066861),
    "left_shoulder_roll_joint": (-0.4363323, 3.1066861),
    "left_shoulder_yaw_joint": (-3.1066861, 3.1066861),
    "left_elbow_joint": (-2.2340214, 0.0),
    "left_wrist_yaw_joint": (-0.5235988, 0.5235988),
    "left_wrist_pitch_joint": (-1.309, 1.309),
    "left_wrist_roll_joint": (-3.1066861, 3.1066861),
    "right_shoulder_pitch_joint": (-3.1066861, 3.1066861),
    "right_shoulder_roll_joint": (-3.1066861, 0.4363323),
    "right_shoulder_yaw_joint": (-3.1066861, 3.1066861),
    "right_elbow_joint": (-2.2340214, 0.0),
    "right_wrist_yaw_joint": (-0.5235988, 0.5235988),
    "right_wrist_pitch_joint": (-1.309, 1.309),
    "right_wrist_roll_joint": (-3.1066861, 3.1066861),
}

coef = 0.1

UPPER_BODY_RANDOM_JOINT_LIMITS = {
    name: (low * coef, high * coef)
    for name, (low, high) in UPPER_BODY_RANDOM_JOINT_LIMITS.items()
}

# ALPHANOID_STAND_JOINT_POS = {
#     ".*_hip_pitch_joint": 0.0,
#     ".*_hip_roll_joint": 0.0,
#     ".*_hip_yaw_joint": 0.0,
#     "left_knee_joint": 0.10,
#     "right_knee_joint": -0.10,
#     ".*_ankle_pitch_joint": 0.0,
#     ".*_ankle_roll_joint": 0.0,
#     "waist.*_joint": 0.0,
#     "head.*_joint": 0.0,
#     ".*_shoulder.*_joint": 0.0,
#     ".*_elbow_joint": 0.0,
#     ".*_wrist.*_joint": 0.0,
# }

ALPHANOID_STAND_JOINT_POS = {
    "left_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.32,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.20,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.32,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist.*_joint": 0.0,
    "head.*_joint": 0.0,
    ".*_shoulder.*_joint": 0.0,
    ".*_elbow_joint": 0.0,
    ".*_wrist.*_joint": 0.0,
}


ALPHANOID_SQUAT_JOINT_POS = {
    "left_hip_pitch_joint": -0.2,
    "right_hip_pitch_joint": 0.2,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.40,
    "right_knee_joint": -0.40,
    ".*_ankle_pitch_joint": 0.2,
    ".*_ankle_roll_joint": 0.0,
    "waist.*_joint": 0.0,
    "head.*_joint": 0.0,
    ".*_shoulder.*_joint": 0.0,
    ".*_elbow_joint": 0.0,
    ".*_wrist.*_joint": 0.0,
}
ALPHANOID_STAND_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, STAND_BASE_HEIGHT),
    joint_pos=ALPHANOID_STAND_JOINT_POS.copy(),
    joint_vel={".*": 0.0},
)
ALPHANOID_SQUAT_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, SQUAT_BASE_HEIGHT),
    joint_pos=ALPHANOID_SQUAT_JOINT_POS.copy(),
    joint_vel={".*": 0.0},
)


ALPHANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ALPHANOID_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ALPHANOID_STAND_INIT_STATE,
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            velocity_limit_sim={
                ".*_hip_pitch_joint": 8.4,
                ".*_hip_roll_joint": 8.4,
                ".*_hip_yaw_joint": 8.4,
                ".*_knee_joint": 8.4,
                ".*_ankle_pitch_joint": 8.4,
                ".*_ankle_roll_joint": 8.4,
            },
            stiffness={
                ".*_hip_pitch_joint": 700.0,
                ".*_hip_roll_joint": 700.0,
                ".*_hip_yaw_joint": 500.0,
                ".*_knee_joint": 700.0,
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 16.8,
            },
            damping={
                ".*_hip_pitch_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_yaw_joint": 5.0,
                ".*_knee_joint": 10.0,
                ".*_ankle_pitch_joint": 2.5,
                ".*_ankle_roll_joint": 1.4,
            },
            effort_limit_sim={
                ".*_hip_pitch_joint": 231.0,
                ".*_hip_roll_joint": 231.0,
                ".*_hip_yaw_joint": 231.0,
                ".*_knee_joint": 231.0,
                ".*_ankle_pitch_joint": 52.5,
                ".*_ankle_roll_joint": 52.5,
            },
            friction={
                ".*_hip_pitch_joint": 2.311,
                ".*_hip_roll_joint": 2.827,
                ".*_hip_yaw_joint": 0.9233,
                ".*_knee_joint": 2.827,
                ".*_ankle_pitch_joint": 0.5666,
                ".*_ankle_roll_joint": 0.5666,
            },
            armature={
                ".*_hip_pitch_joint": 0.01,
                ".*_hip_roll_joint": 0.01,
                ".*_hip_yaw_joint": 0.01,
                ".*_knee_joint": 0.02,
                ".*_ankle_pitch_joint": 0.003,
                ".*_ankle_roll_joint": 0.003,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={"waist_yaw_joint": 140.0},
            stiffness={"waist.*": 500.0},
            damping={"waist.*": 5.0},
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=HEAD_JOINT_NAMES,
            effort_limit_sim={"head.*_joint": 3.36},
            stiffness={".*": 20.0},
            damping={".*": 2.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 135.8,
                ".*_shoulder_roll_joint": 135.8,
                ".*_shoulder_yaw_joint": 71.4,
                ".*_elbow_joint": 75.6,
                ".*_wrist_roll_joint": 46.2,
                ".*_wrist_pitch_joint": 42.0,
                ".*_wrist_yaw_joint": 42.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
                ".*_wrist.*_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 1.5,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_joint": 1.0,
                ".*_wrist.*_joint": 1.0,
            },
        ),
    },
)

def _with_delayed_implicit_actuators(
    robot_cfg: ArticulationCfg,
    min_delay: int = MIN_DELAY_PHY_STEPS,
    max_delay: int = MAX_DELAY_PHY_STEPS,
) -> ArticulationCfg:
    delayed_actuators = {}
    for name, actuator_cfg in robot_cfg.actuators.items():
        delayed_actuators[name] = DelayedImplicitActuatorCfg(
            joint_names_expr=actuator_cfg.joint_names_expr,
            effort_limit=getattr(actuator_cfg, "effort_limit", None),
            effort_limit_sim=getattr(actuator_cfg, "effort_limit_sim", None),
            velocity_limit=getattr(actuator_cfg, "velocity_limit", None),
            velocity_limit_sim=getattr(actuator_cfg, "velocity_limit_sim", None),
            stiffness=actuator_cfg.stiffness,
            damping=actuator_cfg.damping,
            armature=getattr(actuator_cfg, "armature", None),
            friction=getattr(actuator_cfg, "friction", None),
            min_delay=min_delay,
            max_delay=max_delay,
        )
    return robot_cfg.replace(actuators=delayed_actuators)


ALPHANOID_STAND_CFG = ALPHANOID_CFG.replace(init_state=ALPHANOID_STAND_INIT_STATE)
ALPHANOID_STAND_DELAYED_CFG = _with_delayed_implicit_actuators(ALPHANOID_STAND_CFG)
ALPHANOID_SQUAT_CFG = ALPHANOID_CFG.replace(init_state=ALPHANOID_SQUAT_INIT_STATE)
