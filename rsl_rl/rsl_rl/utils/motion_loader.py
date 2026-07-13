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

import glob
import json

import numpy as np
import torch


class AMPLoader:
    # Robot3 lower-body AMP obs (24 dims):
    # right_leg_pos(6) + left_leg_pos(6) + right_leg_vel(6) + left_leg_vel(6)
    JOINT_POS_SIZE = 12
    JOINT_VEL_SIZE = 12

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    # Full-body joint block layout (gmr_data_conversion target_order):
    # [right_arm(7), left_arm(7), waist(1), right_leg(6), left_leg(6)]
    _RIGHT_LEG_POS = slice(15, 21)
    _LEFT_LEG_POS = slice(21, 27)
    _RIGHT_LEG_VEL = slice(42, 48)
    _LEFT_LEG_VEL = slice(48, 54)

    # 66-dim visualization frame offsets
    _VIS_RIGHT_LEG_POS = slice(21, 27)
    _VIS_LEFT_LEG_POS = slice(27, 33)
    _VIS_RIGHT_LEG_VEL = slice(54, 60)
    _VIS_LEFT_LEG_VEL = slice(60, 66)

    @staticmethod
    def extract_amp_obs(motion_data: np.ndarray) -> np.ndarray:
        """Extract 24-dim lower-body AMP obs from full-body motion frames."""
        n_cols = motion_data.shape[1]
        if n_cols == AMPLoader.JOINT_VEL_END_IDX:
            return motion_data

        if n_cols >= 66:
            # [root_pos(3), euler(3), joint_pos(27), lin_vel(3), ang_vel(3), joint_vel(27)]
            right_leg_pos = motion_data[:, AMPLoader._VIS_RIGHT_LEG_POS]
            left_leg_pos = motion_data[:, AMPLoader._VIS_LEFT_LEG_POS]
            right_leg_vel = motion_data[:, AMPLoader._VIS_RIGHT_LEG_VEL]
            left_leg_vel = motion_data[:, AMPLoader._VIS_LEFT_LEG_VEL]
        elif n_cols >= 54:
            # [joint_pos(27), joint_vel(27)]
            right_leg_pos = motion_data[:, AMPLoader._RIGHT_LEG_POS]
            left_leg_pos = motion_data[:, AMPLoader._LEFT_LEG_POS]
            right_leg_vel = motion_data[:, AMPLoader._RIGHT_LEG_VEL]
            left_leg_vel = motion_data[:, AMPLoader._LEFT_LEG_VEL]
        else:
            raise ValueError(
                f"Unsupported motion frame dim={n_cols}. "
                "Expected 24 (legs only), 54 (joint pos+vel), or 66 (visualization)."
            )

        return np.concatenate([right_leg_pos, left_leg_pos, right_leg_vel, left_leg_vel], axis=1)

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        self.device = device
        self.time_between_frames = time_between_frames

        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])
                amp_obs = self.extract_amp_obs(motion_data)

                self.trajectories.append(torch.tensor(amp_obs, dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(amp_obs, dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len:.2f}s lower-body motion ({amp_obs.shape[1]} dims) from {motion_file}.")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions")
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_trajectory(self, traj_idx):
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int64), np.ceil(p * n).astype(np.int64)
        idx_low = np.clip(idx_low, 0, None)
        idx_high = np.clip(idx_high, 0, None)

        amp_dim = AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX
        all_frame_starts = torch.zeros(len(traj_idxs), amp_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), amp_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            idx_low_traj = np.clip(idx_low[traj_mask], 0, trajectory.shape[0] - 1)
            idx_high_traj = np.clip(idx_high[traj_mask], 0, trajectory.shape[0] - 1)
            all_frame_starts[traj_mask] = trajectory[idx_low_traj]
            all_frame_ends[traj_mask] = trajectory[idx_high_traj]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        joints0 = AMPLoader.get_joint_pose(frame0)
        joints1 = AMPLoader.get_joint_pose(frame1)
        joint_vel_0 = AMPLoader.get_joint_vel(frame0)
        joint_vel_1 = AMPLoader.get_joint_vel(frame1)
        return torch.cat([self.slerp(joints0, joints1, blend), self.slerp(joint_vel_0, joint_vel_1, blend)])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_VEL_END_IDX]
                s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_VEL_END_IDX]
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames))
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX : AMPLoader.JOINT_POSE_END_IDX]

    @staticmethod
    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]
