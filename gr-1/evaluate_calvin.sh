#!/bin/bash
EVAL_DIR="/home/jinaoqun/workspace/GR-MG/resources/policy/gr-1/2025-03-10"
MAE_CKPT_PATH="/home/jinaoqun/workspace/GR-MG/resources/MAE/mae_pretrain_vit_base.pth"
POLICY_CKPT_PATH="/home/jinaoqun/workspace/GR-MG/resources/policy/gr-1/2025-03-10/train_policy/epoch=1-step=111632.ckpt"
export CALVIN_ROOT="/home/jinaoqun/workspace/GR-MG/calvin"
export CUDA_VISIBLE_DEVICES=3
export MESA_GL_VERSION_OVERRIDE=3.3

python3 evaluate_calvin.py \
    --eval_dir ${EVAL_DIR} \
    --mae_ckpt_path ${MAE_CKPT_PATH} \
    --policy_ckpt_path ${POLICY_CKPT_PATH} \
    --configs_path config/train.json \
    --dataset_dir task_ABC_D/ \
    ${@:1}
