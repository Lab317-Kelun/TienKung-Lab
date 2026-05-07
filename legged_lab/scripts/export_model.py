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

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export trained policy model to JIT and ONNX formats.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app (headless mode for export)
if args_cli.headless is None:
    args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def export_model():
    """Export trained policy model to JIT and ONNX formats without running simulation."""
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # Minimal environment setup (no need for full simulation)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.physics_material = None
    env_cfg.domain_rand.events.add_base_mass = None
    env_cfg.domain_rand.events.reset_base = None
    env_cfg.domain_rand.events.reset_robot_joints = None
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.randomize_com_displacement = None
    env_cfg.domain_rand.events.randomize_actuator_gains = None
    env_cfg.domain_rand.events.randomize_link_mass = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.action_noise.enable = False
    env_cfg.domain_rand.actuation_offset.enable = False
    env_cfg.scene.num_envs = 1  # Only need 1 environment for export
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Looking for run matching: {agent_cfg.load_run}")

    # Convert checkpoint number to pattern if it's a pure number
    checkpoint_pattern = agent_cfg.load_checkpoint
    if checkpoint_pattern.isdigit():
        # If it's a number, convert to model_{number}.pt pattern
        checkpoint_pattern = f"model_{checkpoint_pattern}.pt"
        print(f"[INFO] Converted checkpoint number to pattern: {checkpoint_pattern}")
    else:
        print(f"[INFO] Looking for checkpoint matching: {checkpoint_pattern}")

    try:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, checkpoint_pattern)
        print(f"[INFO] Found checkpoint: {resume_path}")
    except ValueError as e:
        print(f"[ERROR] Failed to find checkpoint: {e}")
        print(f"[INFO] Available runs in {log_root_path}:")
        if os.path.exists(log_root_path):
            for item in os.listdir(log_root_path):
                item_path = os.path.join(log_root_path, item)
                if os.path.isdir(item_path):
                    print(f"  - {item}")
                    # Check for checkpoints in this run
                    checkpoints = [f for f in os.listdir(item_path) if f.endswith(".pt")]
                    if checkpoints:
                        print(f"    Checkpoints: {len(checkpoints)} files")
                        print(f"    Latest: {sorted(checkpoints)[-1] if checkpoints else 'None'}")
        raise

    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    print(f"\n[INFO] Exporting policy model...")
    print(f"[INFO] Policy device: {env.device}")
    print(f"[INFO] Policy class: {type(runner.alg.policy).__name__}")

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    # Export as JIT (TorchScript)
    jit_path = os.path.join(export_model_dir, "policy.pt")
    print(f"\n[INFO] Exporting JIT model to: {jit_path}")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    print(f"[INFO] ✓ JIT model exported successfully")

    # Export as ONNX
    onnx_path = os.path.join(export_model_dir, "policy.onnx")
    print(f"\n[INFO] Exporting ONNX model to: {onnx_path}")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    print(f"[INFO] ✓ ONNX model exported successfully")

    print(f"\n[INFO] Model export completed!")
    print(f"[INFO] JIT model: {jit_path}")
    print(f"[INFO] ONNX model: {onnx_path}")


if __name__ == "__main__":
    export_model()
    simulation_app.close()
