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

"""AMP expert loader for Robot3 full-body motion (including head).

Training AMP obs (66 dims, body frame):
  [lin_vel(3), ang_vel(3), joint_pos(30), joint_vel(30)]

Joint order:
  right_arm(7), left_arm(7), waist(1), right_leg(6), left_leg(6), head(3)

Also accepts 72-dim GMR visualization frames and converts them to 66-dim AMP obs.
"""

import glob
import json

import numpy as np
import torch


class AMPLoader:
    # AMP obs layout (full body including head)
    ROOT_LIN_VEL_SIZE = 3
    ROOT_ANG_VEL_SIZE = 3
    ROOT_VEL_SIZE = ROOT_LIN_VEL_SIZE + ROOT_ANG_VEL_SIZE  # 6
    JOINT_POS_SIZE = 30
    JOINT_VEL_SIZE = 30
    AMP_OBS_SIZE = ROOT_VEL_SIZE + JOINT_POS_SIZE + JOINT_VEL_SIZE  # 66

    ROOT_LIN_VEL_START_IDX = 0
    ROOT_LIN_VEL_END_IDX = ROOT_LIN_VEL_START_IDX + ROOT_LIN_VEL_SIZE
    ROOT_ANG_VEL_START_IDX = ROOT_LIN_VEL_END_IDX
    ROOT_ANG_VEL_END_IDX = ROOT_ANG_VEL_START_IDX + ROOT_ANG_VEL_SIZE

    JOINT_POSE_START_IDX = ROOT_ANG_VEL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    # 72 = official GMR visualization with head
    # 66 = AMP (with head) after play_amp; legacy vis-without-head also 66 (detected by root z)
    # 60 = legacy AMP without head (pad head zeros)
    _VIS_DIM_WITH_HEAD = 72
    _LEGACY_AMP_NO_HEAD = 60

    @staticmethod
    def _gmr_joints_to_amp(joint_block: np.ndarray) -> np.ndarray:
        """Reorder GMR joints → AMP joint order (keeps head if present).

        GMR: Lleg6, Rleg6, waist1, Larm7, Rarm7 [, head3]
        AMP: Rarm7, Larm7, waist1, Rleg6, Lleg6 [, head3]
        """
        left_leg = joint_block[:, 0:6]
        right_leg = joint_block[:, 6:12]
        waist = joint_block[:, 12:13]
        left_arm = joint_block[:, 13:20]
        right_arm = joint_block[:, 20:27]
        parts = [right_arm, left_arm, waist, right_leg, left_leg]
        if joint_block.shape[1] >= 30:
            parts.append(joint_block[:, 27:30])
        return np.concatenate(parts, axis=1)

    @staticmethod
    def extract_amp_obs(motion_data: np.ndarray, lin_vel_frame: str = "world") -> np.ndarray:
        """Convert frames to 66-dim AMP obs (full body including head)."""
        n_cols = motion_data.shape[1]
        n = motion_data.shape[0]

        if n_cols == AMPLoader.AMP_OBS_SIZE:
            # AMP-66 vs legacy vis-66 (no head): vis root z ≈ 1m at index 2
            if abs(float(motion_data[:, 2].mean())) > 0.3:
                from scipy.spatial.transform import Rotation

                euler = motion_data[:, 3:6]
                joint_pos = motion_data[:, 6:33]
                lin = motion_data[:, 33:36]
                ang = motion_data[:, 36:39]
                joint_vel = motion_data[:, 39:66]
                if lin_vel_frame == "world":
                    lin = Rotation.from_euler("XYZ", euler).apply(lin, inverse=True)
                zeros3 = np.zeros((n, 3), dtype=np.float64)
                joint_pos = np.concatenate([AMPLoader._gmr_joints_to_amp(joint_pos), zeros3], axis=1)
                joint_vel = np.concatenate([AMPLoader._gmr_joints_to_amp(joint_vel), zeros3], axis=1)
                return np.concatenate([lin, ang, joint_pos, joint_vel], axis=1)
            return motion_data.astype(np.float64, copy=False)

        if n_cols == AMPLoader._LEGACY_AMP_NO_HEAD:
            zeros3 = np.zeros((n, 3), dtype=np.float64)
            return np.concatenate(
                [motion_data[:, :6], motion_data[:, 6:33], zeros3, motion_data[:, 33:60], zeros3],
                axis=1,
            )

        from scipy.spatial.transform import Rotation

        if n_cols >= AMPLoader._VIS_DIM_WITH_HEAD:
            n_j = 30
            euler = motion_data[:, 3:6]
            joint_pos = motion_data[:, 6 : 6 + n_j]
            lin = motion_data[:, 6 + n_j : 9 + n_j]
            ang = motion_data[:, 9 + n_j : 12 + n_j]
            joint_vel = motion_data[:, 12 + n_j : 12 + 2 * n_j]
            if lin_vel_frame == "world":
                lin = Rotation.from_euler("XYZ", euler).apply(lin, inverse=True)
            elif lin_vel_frame != "body":
                raise ValueError(f"Unknown lin_vel_frame={lin_vel_frame}")
            joint_pos = AMPLoader._gmr_joints_to_amp(joint_pos)
            joint_vel = AMPLoader._gmr_joints_to_amp(joint_vel)
            return np.concatenate([lin, ang, joint_pos, joint_vel], axis=1)

        raise ValueError(
            f"Unsupported motion frame dim={n_cols}. "
            f"Expected {AMPLoader.AMP_OBS_SIZE} (AMP with head), "
            f"{AMPLoader._LEGACY_AMP_NO_HEAD} (legacy AMP), "
            f"or {AMPLoader._VIS_DIM_WITH_HEAD} (GMR visualization)."
        )

    def __init__(
        self,
        device,
        time_between_frames,
        data_dir="",
        preload_transitions=False,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
        obs_indices=None,
        lin_vel_frame: str = "world",
    ):
        """Expert dataset for AMP discriminator / reward.

        time_between_frames: seconds between (s, s_next) transitions (usually env.step_dt).
        lin_vel_frame: used only when loading 66-dim visualization files.
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
                amp_obs = AMPLoader.extract_amp_obs(motion_data, lin_vel_frame=lin_vel_frame)

                self.trajectories.append(torch.tensor(amp_obs, dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(amp_obs, dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (amp_obs.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(amp_obs.shape[0]))

            print(
                f"Loaded {traj_len:.2f}s AMP motion ({amp_obs.shape[1]} dims: "
                f"lin/ang vel + joint_pos/vel) from {motion_file}."
            )

        self._full_obs_dim = self.trajectories[0].shape[1]
        if self._full_obs_dim != self.AMP_OBS_SIZE:
            raise ValueError(f"Expected AMP obs dim={self.AMP_OBS_SIZE}, got {self._full_obs_dim}")

        self._obs_indices = None
        if obs_indices is not None:
            obs_indices = [int(idx) for idx in obs_indices]
            if len(obs_indices) == 0:
                raise ValueError("obs_indices must contain at least one index.")
            if min(obs_indices) < 0 or max(obs_indices) >= self._full_obs_dim:
                raise ValueError("obs_indices contain out-of-range values.")
            self._obs_indices = obs_indices

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
        all_frame_starts = torch.zeros(len(traj_idxs), self._full_obs_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self._full_obs_dim, device=self.device)
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
        """Yields batches of (s, s_next) AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs]
                s_next = self.preloaded_s_next[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                s = self.get_frame_at_time_batch(traj_idxs, times)
                s_next = self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

            if self._obs_indices is not None:
                s = s[:, self._obs_indices]
                s_next = s_next[:, self._obs_indices]
            yield s, s_next

    @property
    def observation_dim(self):
        if self._obs_indices is None:
            return self._full_obs_dim
        return len(self._obs_indices)

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    @staticmethod
    def get_root_lin_vel(pose):
        return pose[AMPLoader.ROOT_LIN_VEL_START_IDX : AMPLoader.ROOT_LIN_VEL_END_IDX]

    @staticmethod
    def get_root_ang_vel(pose):
        return pose[AMPLoader.ROOT_ANG_VEL_START_IDX : AMPLoader.ROOT_ANG_VEL_END_IDX]

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
