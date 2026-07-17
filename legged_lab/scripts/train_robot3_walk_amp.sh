#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=5
export WANDB_ENTITY=polar-bear

python legged_lab/scripts/train.py \
  --task=robot3_walk_amp \
  --headless \
  --num_envs=4096 \
  --logger=wandb \
  --seed=42 \
  --run_name=lower_nowaist_newwalk1
