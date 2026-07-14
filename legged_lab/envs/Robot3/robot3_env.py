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

"""Robot3 lower-body environment (13-DOF policy: waist + legs).

Upper body (arms + head) is PD-locked at default pose.
AMP obs: 26 dims = joint_pos(13) + joint_vel(13).
"""

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.Robot3.walk_cfg import Robot3WalkAmpFlatEnvCfg, Robot3WalkFlatEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class Robot3Env(VecEnv):
    def __init__(
        self,
        cfg: Robot3WalkFlatEnvCfg | Robot3WalkAmpFlatEnvCfg,
        headless,
    ):
        self.cfg: Robot3WalkFlatEnvCfg | Robot3WalkAmpFlatEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True
        )
        # Isaac joint indices (for sim write / visualize / AMP from state)
        self.left_leg_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_joint_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            preserve_order=True,
        )
        self.waist_joint_ids, _ = self.robot.find_joints(name_keys=["waist_yaw_joint"], preserve_order=True)
        self.head_ids, _ = self.robot.find_joints(
            name_keys=["head_yaw_joint", "head_roll_joint", "head_pitch_joint"],
            preserve_order=True,
        )

        # Policy / AMP order (13): waist(1), right_leg(6), left_leg(6)
        self.controlled_joint_ids = (
            list(self.waist_joint_ids) + list(self.right_leg_joint_ids) + list(self.left_leg_joint_ids)
        )
        self.upper_body_joint_ids = list(self.left_arm_ids) + list(self.right_arm_ids) + list(self.head_ids)
        self.num_actions = len(self.controlled_joint_ids)  # 13

        # Action-local indices (for rewards that index into env.action)
        self.waist_ids = [0]
        self.right_leg_ids = list(range(1, 7))
        self.left_leg_ids = list(range(7, 13))
        self.ankle_joint_ids = [
            self.left_leg_ids[4],
            self.right_leg_ids[4],
            self.left_leg_ids[5],
            self.right_leg_ids[5],
        ]

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.25], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.25], device=self.device).repeat((self.num_envs, 1))

        # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        # For feet_clearance reward: cumulative foot height (similar to Gym version)
        self.feet_height = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_feet_z = torch.full(
            (self.num_envs, 2), 0.05, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

        # Official GMR / gmr_data_conversion joint order (with head):
        # [root_pos(3), euler(3),
        #  left_leg(6), right_leg(6), waist(1), left_arm(7), right_arm(7), head(3),
        #  lin_vel_w(3), ang_vel_b(3),
        #  left_leg_vel(6), right_leg_vel(6), waist_vel(1), left_arm_vel(7), right_arm_vel(7), head_vel(3)]
        # Total = 72
        n_left_leg = len(self.left_leg_ids)
        n_right_leg = len(self.right_leg_ids)
        n_waist = len(self.waist_ids)
        n_left_arm = len(self.left_arm_ids)
        n_right_arm = len(self.right_arm_ids)
        n_head = len(self.head_ids)
        n_joint_gmr = n_left_leg + n_right_leg + n_waist + n_left_arm + n_right_arm + n_head
        required_dim = 12 + 2 * n_joint_gmr
        if visual_motion_frame.shape[0] < required_dim:
            raise ValueError(
                f"visual_motion_frame dim={visual_motion_frame.shape[0]} < required {required_dim} "
                f"(GMR layout with {n_joint_gmr} joints). Got wrong visualization file?"
            )

        i = 6
        left_leg_pos = visual_motion_frame[i : i + n_left_leg]
        i += n_left_leg
        right_leg_pos = visual_motion_frame[i : i + n_right_leg]
        i += n_right_leg
        waist_pos = visual_motion_frame[i : i + n_waist]
        i += n_waist
        left_arm_pos = visual_motion_frame[i : i + n_left_arm]
        i += n_left_arm
        right_arm_pos = visual_motion_frame[i : i + n_right_arm]
        i += n_right_arm
        head_pos = visual_motion_frame[i : i + n_head]
        i += n_head

        lin_vel_w = visual_motion_frame[i : i + 3].clone()
        i += 3
        ang_vel_b = visual_motion_frame[i : i + 3].clone()
        i += 3

        left_leg_vel = visual_motion_frame[i : i + n_left_leg]
        i += n_left_leg
        right_leg_vel = visual_motion_frame[i : i + n_right_leg]
        i += n_right_leg
        waist_vel = visual_motion_frame[i : i + n_waist]
        i += n_waist
        left_arm_vel = visual_motion_frame[i : i + n_left_arm]
        i += n_left_arm
        right_arm_vel = visual_motion_frame[i : i + n_right_arm]
        i += n_right_arm
        head_vel = visual_motion_frame[i : i + n_head]

        dof_pos[:, self.left_leg_joint_ids] = left_leg_pos
        dof_pos[:, self.right_leg_joint_ids] = right_leg_pos
        dof_pos[:, self.waist_joint_ids] = waist_pos
        dof_pos[:, self.left_arm_ids] = left_arm_pos
        dof_pos[:, self.right_arm_ids] = right_arm_pos
        dof_pos[:, self.head_ids] = head_pos

        dof_vel[:, self.left_leg_joint_ids] = left_leg_vel
        dof_vel[:, self.right_leg_joint_ids] = right_leg_vel
        dof_vel[:, self.waist_joint_ids] = waist_vel
        dof_vel[:, self.left_arm_ids] = left_arm_vel
        dof_vel[:, self.right_arm_ids] = right_arm_vel
        dof_vel[:, self.head_ids] = head_vel

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        # Keep mocap height; do not offset (offset polluted root velocity after physics step).
        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        # Official gmr: lin=WORLD, ang=BODY. Isaac root_state wants WORLD.
        ang_vel_w = quat_apply(quat_wxyz.unsqueeze(0), ang_vel_b.unsqueeze(0)).squeeze(0)

        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel_w.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel_w.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        # Export AMP obs: lower body only (26): waist, right_leg, left_leg (+ vels)
        amp_obs = torch.cat(
            (
                waist_pos,
                right_leg_pos,
                left_leg_pos,
                waist_vel,
                right_leg_vel,
                left_leg_vel,
            ),
            dim=-1,
        )
        return amp_obs.unsqueeze(0).expand(self.num_envs, -1).contiguous()

    def _compute_hand_positions(self):
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        return left_hand_pos, right_hand_pos

    def _compute_foot_positions(self):
        left_foot_pos = self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        right_foot_pos = self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        return left_foot_pos, right_foot_pos

    def _build_amp_obs_from_state(self):
        """Build AMP obs (26 dims): waist + legs joint pos/vel only."""
        waist_dof_pos = self.robot.data.joint_pos[:, self.waist_joint_ids]
        right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_joint_ids]
        left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_joint_ids]
        waist_dof_vel = self.robot.data.joint_vel[:, self.waist_joint_ids]
        right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_joint_ids]
        left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_joint_ids]

        return torch.cat(
            (
                waist_dof_pos,
                right_leg_dof_pos,
                left_leg_dof_pos,
                waist_dof_vel,
                right_leg_dof_vel,
                left_leg_dof_vel,
            ),
            dim=-1,
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.controlled_joint_ids] - robot.data.default_joint_pos[:, self.controlled_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.controlled_joint_ids] - robot.data.default_joint_vel[:, self.controlled_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 13
                joint_vel * self.obs_scales.joint_vel,  # 13
                action * self.obs_scales.actions,  # 13
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # Append the flattened depth data to the end of the actor and critic observation vectors.
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0
        # Reset feet clearance state
        self.feet_height[env_ids] = 0.0
        self.last_feet_z[env_ids] = 0.05

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # 添加动作噪声（域随机化）
        if self.cfg.domain_rand.action_noise.enable:
            self.action += self.cfg.domain_rand.action_noise.noise_scale * torch.randn_like(self.action) * self.action

        # Lower body from policy; upper body locked at default pose (PD hold).
        processed_actions = self.robot.data.default_joint_pos.clone()
        processed_actions[:, self.controlled_joint_ids] = (
            self.action * self.action_scale + self.robot.data.default_joint_pos[:, self.controlled_joint_ids]
        )
        processed_actions[:, self.upper_body_joint_ids] = self.robot.data.default_joint_pos[
            :, self.upper_body_joint_ids
        ]

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        reset_buf = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        # projected_gravity = self.robot.data.projected_gravity_b
        # gravity_termination_buf = torch.any(torch.norm(projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.7, dim=1)
        # base_height = self.robot.data.root_pos_w[:, 2]
        # height_cutoff = base_height > 1.5
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        # reset_buf |= gravity_termination_buf
        # reset_buf |= height_cutoff
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # Actor obs: [ang_vel(3), gravity(3), command(3), joint_pos(N), joint_vel(N), action(N)]
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang_vel (角速度)
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity  # projected_gravity
            noise_vec[6:9] = 0.0  # command (命令不需要噪声)
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos  # joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = noise_scales.joint_vel * self.obs_scales.joint_vel  # joint_vel
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0  # action (动作不需要噪声)
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras
 
    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs_for_expert_trans(self):
        """Gets AMP obs: waist + legs pos/vel (26 dims)."""
        return self._build_amp_obs_from_state()


    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
