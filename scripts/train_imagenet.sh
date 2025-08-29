#!/usr/bin/env bash

set -euo pipefail

IMAGENET_ROOT=${1:-/workspace/imagenet-1k}
EXP_NAME=${2:-resolution_224}
EXTRA_ARGS=("${@:3}")

# 解析脚本与工程根路径
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 选择存在的训练与验证目录
if [ -d "$IMAGENET_ROOT/train" ]; then
  TRAIN_PATH="$IMAGENET_ROOT/train"
else
  TRAIN_PATH="$IMAGENET_ROOT"
fi
if [ -d "$IMAGENET_ROOT/val" ]; then
  VAL_PATH="$IMAGENET_ROOT/val"
else
  VAL_PATH=""
fi

echo "Using train path: $TRAIN_PATH"
if [ -n "$VAL_PATH" ]; then
  echo "Using val path: $VAL_PATH"
fi



CUDA_VISIBLE_DEVICES=3 python -m stegastamp.train "$EXP_NAME" \
  --train_path "$TRAIN_PATH" \
  ${VAL_PATH:+--val_path "$VAL_PATH"} \
  --height 224 --width 224 \
  --secret_size 100 \
  --num_steps 100000 \
  --lr 1e-4 \
  --batch_size 32 \
  --rnd_trans 0.1 --rnd_noise 0.02 --rnd_bri 0.3 --rnd_sat 1.0 --rnd_hue 0.1 \
  --contrast_low 0.5 --contrast_high 1.5 \
  --jpeg_quality 50 \
  --output_root "$ROOT_DIR/output" \
  --run_script "$SCRIPT_PATH" \
  --log_interval 100 \
  --save_interval 1000 \
  --val_interval 10000 \
  --no_im_loss_steps 20000 \
  --rnd_trans_ramp 10000 \
  --l2_loss_ramp 15000 \
  --lpips_loss_ramp 15000 \
  --G_loss_ramp 15000 \
  --secret_loss_scale 1.5 \
  --l2_loss_scale 2 \
  --lpips_loss_scale 1.5 \
  --G_loss_scale 0.5 \
  --y_scale 1 \
  --u_scale 100 \
  --v_scale 100 \
  --l2_edge_gain 10 \
  --l2_edge_ramp 10000 \
  --l2_edge_delay 80000 \
  "${EXTRA_ARGS[@]}"


