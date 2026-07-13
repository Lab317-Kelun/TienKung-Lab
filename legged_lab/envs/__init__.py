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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.envs.robot1_6.robot1_6_env import Robot1_6Env
from legged_lab.envs.robot1_6.run_cfg import Robot1_6RunAgentCfg, Robot1_6RunFlatEnvCfg
from legged_lab.envs.robot1_6.run_with_sensor_cfg import (
    Robot1_6RunWithSensorAgentCfg,
    Robot1_6RunWithSensorFlatEnvCfg,
)
from legged_lab.envs.robot1_6.walk_cfg import (
    Robot1_6WalkAgentCfg,
    Robot1_6WalkAmpAgentCfg,
    Robot1_6WalkAmpFlatEnvCfg,
    Robot1_6WalkFlatEnvCfg,
)
from legged_lab.envs.robot1_6.walk_with_sensor_cfg import (
    Robot1_6WalkWithSensorAgentCfg,
    Robot1_6WalkWithSensorFlatEnvCfg,
)

from legged_lab.envs.Robot3.robot3_env import Robot3Env
from legged_lab.envs.Robot3.walk_cfg import (
    Robot3WalkAgentCfg,
    Robot3WalkAmpAgentCfg,
    Robot3WalkAmpFlatEnvCfg,
    Robot3WalkFlatEnvCfg,
)

# from legged_lab.envs.robot1_6.velocity_cfg import (
#     Robot1_6VelocityAgentCfg,
#     Robot1_6VelocityEnvCfg,
# )
from legged_lab.utils.task_registry import task_registry

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)

# Robot1_6 tasks (lower body only, 12 DOF)
task_registry.register("robot1_6_walk", Robot1_6Env, Robot1_6WalkFlatEnvCfg(), Robot1_6WalkAgentCfg())
task_registry.register("robot1_6_walk_amp", Robot1_6Env, Robot1_6WalkAmpFlatEnvCfg(), Robot1_6WalkAmpAgentCfg())
task_registry.register("robot1_6_run", Robot1_6Env, Robot1_6RunFlatEnvCfg(), Robot1_6RunAgentCfg())
task_registry.register(
    "robot1_6_walk_with_sensor", Robot1_6Env, Robot1_6WalkWithSensorFlatEnvCfg(), Robot1_6WalkWithSensorAgentCfg()
)
task_registry.register(
    "robot1_6_run_with_sensor", Robot1_6Env, Robot1_6RunWithSensorFlatEnvCfg(), Robot1_6RunWithSensorAgentCfg()
)
# Robot1_6 velocity task (AMP + RL, only feet_gait reward)
# task_registry.register("robot1_6_velocity", Robot1_6Env, Robot1_6VelocityEnvCfg(), Robot1_6VelocityAgentCfg())

# Robot3 tasks (12-DOF lower body policy control)
task_registry.register("robot3_walk_amp", Robot3Env, Robot3WalkAmpFlatEnvCfg(), Robot3WalkAmpAgentCfg())
