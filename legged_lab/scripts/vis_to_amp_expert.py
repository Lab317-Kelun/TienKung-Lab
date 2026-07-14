#!/usr/bin/env python3
"""Convert GMR visualization (72) → 32-dim AMP expert (lower body + root vel).

AMP layout:
  [lin_vel_b(3), ang_vel_b(3),
   waist(1), right_leg(6), left_leg(6),
   same for joint_vel]
Total = 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rsl_rl.utils.motion_loader import AMPLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lin-frame", choices=["world", "body"], default="world")
    args = p.parse_args()

    with open(args.vis) as f:
        data = json.load(f)
    frames = np.asarray(data["Frames"], dtype=np.float64)
    amp = AMPLoader.extract_amp_obs(frames, lin_vel_frame=args.lin_frame)
    assert amp.shape[1] == 32, amp.shape

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {data.get("FrameDuration", 0.033)},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write(f'"MotionWeight": {data.get("MotionWeight", 0.5)},\n\n')
        f.write('"Frames":\n[\n')
        for i, row in enumerate(amp):
            line = ", ".join(f"{x:.6f}" for x in row)
            end = "]\n" if i == len(amp) - 1 else "],\n"
            f.write("  [" + line + end)
        f.write("]\n}")

    print(f"Wrote {args.out} shape={amp.shape}")
    print(f"  mean lin_b={amp[:, :3].mean(0).round(4)}")
    print(f"  mean ang_b={amp[:, 3:6].mean(0).round(4)}")


if __name__ == "__main__":
    main()
