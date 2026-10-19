#!/usr/bin/env bash
# 统一环境变量。用法：source scripts/env.sh
set -a

# Hugging Face 直连不稳定，统一走镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/xht/MLwork/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# SimplerEnv / ManiSkill2 渲染：默认 Vulkan loader 会选到 lavapipe，必须强制 NVIDIA ICD
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=""

# 使用 GPU 0；另外两张卡保持空闲，方便实验室其他工作
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# mediapy 需要 PATH 里有 ffmpeg（venv 里已链接 imageio-ffmpeg 的二进制）
export PATH="/home/xht/MLwork/venv_openvla/bin:$PATH"

set +a
