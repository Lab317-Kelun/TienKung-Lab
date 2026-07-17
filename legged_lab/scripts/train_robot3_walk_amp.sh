#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4
export WANDB_ENTITY=polar-bear

python legged_lab/scripts/train.py \
  --task=robot3_walk_amp \
  --headless \
  --num_envs=4096 \
  --logger=wandb \
  --seed=42 \
  --run_name=lower_nowaist_stand2 \
  --resume True \
  --load_run 2026-07-17_02-43-12_lower_nowaist_walk1 \
  --checkpoint model_.*.pt \