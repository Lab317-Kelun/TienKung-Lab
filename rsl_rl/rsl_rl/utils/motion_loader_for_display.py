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

"""Visualization motion loader for Robot3.

Official gmr_data_conversion (with head) writes 72-dim frames:
  [root_pos(3), euler_XYZ(3),
   left_leg(6), right_leg(6), waist(1), left_arm(7), right_arm(7), head(3),   # 30
   lin_vel_w(3), ang_vel_b(3),
   left_leg_vel(6), ..., head_vel(3)]                                           # 30
Total = 72

lin_vel = world-frame finite difference
ang_vel = body-frame relative rotation / dt
"""

import glob
import json

import numpy as np
import torch


class AMPLoaderDisplay:
    # Full GMR visualization frame with head joints
    FRAME_SIZE = 72
    JOINT_POS_SIZE = 30  # includes head(3)
    JOINT_VEL_SIZE = 30

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = 3
    ROOT_EULER_START_IDX = 3
    ROOT_EULER_END_IDX = 6
    JOINT_POSE_START_IDX = 6
    JOINT_POSE_END_IDX = 36
    ROOT_LIN_VEL_START_IDX = 36
    ROOT_LIN_VEL_END_IDX = 39
    ROOT_ANG_VEL_START_IDX = 39
    ROOT_ANG_VEL_END_IDX = 42
    JOINT_VEL_START_IDX = 42
    JOINT_VEL_END_IDX = 72

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Load visualization frames for play_amp / visualize_motion.

        Real path comes from ``env_cfg.amp_motion_files_display``.
        AMP training experts are loaded by ``AMPLoader`` from ``amp_motion_files``.
        """
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
                motion_data = np.array(motion_json["Frames"], dtype=np.float64)
                if motion_data.shape[1] < self.FRAME_SIZE:
                    raise ValueError(
                        f"{motion_file}: expected >= {self.FRAME_SIZE} dims "
                        f"(GMR visualization with head), got {motion_data.shape[1]}"
                    )
                # Keep full GMR frame (do NOT truncate — that scrambled joints/vels).
                frame = motion_data[:, : self.FRAME_SIZE]
                self.trajectories.append(torch.tensor(frame, dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(frame, dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (frame.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(frame.shape[0]))

            print(
                f"Loaded {traj_len:.2f}s visualization motion ({frame.shape[1]} dims) from {motion_file}."
            )

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
            self.preloaded_s_next = self.get_full_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
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
        idx_low = min(idx_low, n - 1)
        idx_high = min(idx_high, n - 1)
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.clip(np.floor(p * n).astype(np.int64), 0, None)
        idx_high = np.clip(np.ceil(p * n).astype(np.int64), 0, None)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            low = np.clip(idx_low[traj_mask], 0, trajectory.shape[0] - 1)
            high = np.clip(idx_high[traj_mask], 0, trajectory.shape[0] - 1)
            all_frame_starts[traj_mask] = trajectory[low]
            all_frame_ends[traj_mask] = trajectory[high]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        return self.get_frame_at_time(traj_idx, time)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        return self.get_frame_at_time_batch(traj_idxs, times)

    def get_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        return self.get_frame_at_time(traj_idx, self.traj_time_sample(traj_idx))

    def get_full_frame(self):
        return self.get_frame()

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                yield (
                    self.get_frame_at_time_batch(traj_idxs, times),
                    self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames),
                )

    @property
    def observation_dim(self):
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)
