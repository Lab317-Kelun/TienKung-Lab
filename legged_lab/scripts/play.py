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
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    # 禁用所有域随机化
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
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.5, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (-0.0, 0.0)
    # env_cfg.commands.ranges.heading = (-3.14, 3.14)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    # # 打印 Isaac Lab 中的实际关节顺序
    # robot = env.robot
    # isaac_joint_names = robot.data.joint_names
    # print("\n" + "=" * 80)
    # print("Isaac Lab 实际关节顺序 (robot.data.joint_names):")
    # print("=" * 80)
    # for i, name in enumerate(isaac_joint_names):
    #     print(f"  [{i:2d}] {name}")
    # print("=" * 80 + "\n")

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
                    checkpoints = [f for f in os.listdir(item_path) if f.endswith('.pt')]
                    if checkpoints:
                        print(f"    Checkpoints: {len(checkpoints)} files")
                        print(f"    Latest: {sorted(checkpoints)[-1] if checkpoints else 'None'}")
        raise
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    
    # 用于控制打印频率
    step_count = 0
    print_interval = 10  # 每50步打印一次

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            
            # 打印命令速度和实际速度
            step_count += 1
            if step_count % print_interval == 0:
                # 获取命令速度（第一个环境）
                command_vel = env.command_generator.command[0].cpu().numpy()
                # 获取实际速度（body frame，第一个环境）
                actual_vel_body = env.robot.data.root_lin_vel_b[0].cpu().numpy()
                # 获取实际角速度（body frame，第一个环境）
                actual_ang_vel_body = env.robot.data.root_ang_vel_b[0].cpu().numpy()
                # 获取实际速度（world frame，第一个环境）
                actual_vel_world = env.robot.data.root_lin_vel_w[0].cpu().numpy()
                # 获取 base_link 高度（world frame，第一个环境）
                base_height = env.robot.data.root_pos_w[0, 2].cpu().numpy()
                
                # 获取脚部位置并计算距离
                left_foot_pos_w = env.robot.data.body_pos_w[0, env.feet_body_ids[0], :].cpu().numpy()
                right_foot_pos_w = env.robot.data.body_pos_w[0, env.feet_body_ids[1], :].cpu().numpy()
                
                # 计算脚部在body frame中的位置
                root_pos_w = env.robot.data.root_link_pos_w[0, :].cpu().numpy()
                left_foot_pos_rel = left_foot_pos_w - root_pos_w
                right_foot_pos_rel = right_foot_pos_w - root_pos_w
                
                from isaaclab.utils.math import quat_apply, quat_conjugate
                root_quat = env.robot.data.root_link_quat_w[0:1, :].cpu()
                left_foot_pos_rel_t = torch.from_numpy(left_foot_pos_rel).unsqueeze(0).to(root_quat.device)
                right_foot_pos_rel_t = torch.from_numpy(right_foot_pos_rel).unsqueeze(0).to(root_quat.device)
                
                left_foot_pos_b = quat_apply(quat_conjugate(root_quat), left_foot_pos_rel_t)[0].cpu().numpy()
                right_foot_pos_b = quat_apply(quat_conjugate(root_quat), right_foot_pos_rel_t)[0].cpu().numpy()
                
                feet_lateral_distance = abs(left_foot_pos_b[1] - right_foot_pos_b[1])
                
                # 获取膝盖位置并计算距离
                knee_body_ids, _ = env.robot.find_bodies(
                    name_keys=["left_knee_link", "right_knee_link"],
                    preserve_order=True,
                )
                left_knee_pos_w = env.robot.data.body_pos_w[0, knee_body_ids[0], :].cpu().numpy()
                right_knee_pos_w = env.robot.data.body_pos_w[0, knee_body_ids[1], :].cpu().numpy()
                
                left_knee_pos_rel = left_knee_pos_w - root_pos_w
                right_knee_pos_rel = right_knee_pos_w - root_pos_w
                
                left_knee_pos_rel_t = torch.from_numpy(left_knee_pos_rel).unsqueeze(0).to(root_quat.device)
                right_knee_pos_rel_t = torch.from_numpy(right_knee_pos_rel).unsqueeze(0).to(root_quat.device)
                
                left_knee_pos_b = quat_apply(quat_conjugate(root_quat), left_knee_pos_rel_t)[0].cpu().numpy()
                right_knee_pos_b = quat_apply(quat_conjugate(root_quat), right_knee_pos_rel_t)[0].cpu().numpy()
                
                knee_lateral_distance = abs(left_knee_pos_b[1] - right_knee_pos_b[1])
                
                # 计算速度误差
                vel_error_x = abs(command_vel[0] - actual_vel_body[0])
                vel_error_y = abs(command_vel[1] - actual_vel_body[1])
                ang_vel_error_z = abs(command_vel[2] - actual_ang_vel_body[2])
                
                print(f"\n{'='*60}")
                print(f"[Step {step_count}] 速度信息 (环境 0)")
                print(f"{'-'*60}")
                print(f"命令速度 (body frame):")
                print(f"  lin_vel_x = {command_vel[0]:7.4f} m/s")
                print(f"  lin_vel_y = {command_vel[1]:7.4f} m/s")
                print(f"  ang_vel_z = {command_vel[2]:7.4f} rad/s")
                print(f"实际速度 (body frame):")
                print(f"  lin_vel_x = {actual_vel_body[0]:7.4f} m/s  (误差: {vel_error_x:.4f})")
                print(f"  lin_vel_y = {actual_vel_body[1]:7.4f} m/s  (误差: {vel_error_y:.4f})")
                print(f"  lin_vel_z = {actual_vel_body[2]:7.4f} m/s")
                print(f"  ang_vel_z = {actual_ang_vel_body[2]:7.4f} rad/s  (误差: {ang_vel_error_z:.4f})")
                print(f"实际速度 (world frame):")
                print(f"  lin_vel_x = {actual_vel_world[0]:7.4f} m/s")
                print(f"  lin_vel_y = {actual_vel_world[1]:7.4f} m/s")
                print(f"  lin_vel_z = {actual_vel_world[2]:7.4f} m/s")
                print(f"Base_link 高度:")
                print(f"  height_z  = {base_height:7.4f} m")
                print(f"{'-'*60}")
                print(f"脚部和膝盖距离 (body frame):")
                print(f"  脚部横向距离 = {feet_lateral_distance:7.4f} m ({feet_lateral_distance*100:5.1f} cm)")
                print(f"  膝盖横向距离 = {knee_lateral_distance:7.4f} m ({knee_lateral_distance*100:5.1f} cm)")
                print(f"{'='*60}")


if __name__ == "__main__":
    play()
    simulation_app.close()
