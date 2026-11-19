#!/usr/bin/env bash
# ==============================================================================
# Script: run_gsm8k_sft.sh
# Purpose: Supervised Fine-Tuning (SFT) on GSM8K using veRL with PyTorch FSDP
# Target: GPU node on ASU Sol Supercomputer (or any Linux multi-GPU machine)
# ==============================================================================

set -euo pipefail

# 1. Configuration & Default Paths
NUM_GPUS="${1:-2}"
MODEL_PATH="${2:-Qwen/Qwen2.5-1.5B}"

# Shift off positional arguments so they aren't forwarded to Hydra
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi

if [ -d "/scratch/$USER" ]; then
    BASE_DIR="/scratch/$USER"
else
    BASE_DIR="$HOME"
fi

DATA_DIR="${DATA_DIR:-$BASE_DIR/data/gsm8k_sft}"
SAVE_PATH="${SAVE_PATH:-$BASE_DIR/checkpoints/qwen2.5-1.5b-gsm8k-sft}"
EPOCHS="${EPOCHS:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LR="${LR:-2e-5}"
PROJECT_NAME="${PROJECT_NAME:-gsm8k-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_1_5b_sft}"

# 2. Redirect Cache Away From Home Quota & Clean GPU Environment
export HF_HOME="${BASE_DIR}/hf_cache"
export TMPDIR="${BASE_DIR}/tmp"
mkdir -p "${HF_HOME}" "${TMPDIR}" "${SAVE_PATH}"

# Clear AMD ROCm variables that SLURM sets on heterogeneous clusters
unset ROCR_VISIBLE_DEVICES || true
unset HIP_VISIBLE_DEVICES || true

# Logger: Use wandb if API key is present, otherwise console only
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGER='["console","wandb"]'
else
    LOGGER='["console"]'
fi

echo "=============================================================================="
echo " Starting veRL GSM8K Supervised Fine-Tuning"
echo "------------------------------------------------------------------------------"
echo " Model:             ${MODEL_PATH}"
echo " Number of GPUs:    ${NUM_GPUS}"
echo " Training Data:     ${DATA_DIR}/train.parquet"
echo " Validation Data:   ${DATA_DIR}/test.parquet"
echo " Checkpoint Dir:    ${SAVE_PATH}"
echo " Epochs:            ${EPOCHS}"
echo " Micro Batch Size:  ${MICRO_BATCH_SIZE} per GPU"
echo " Learning Rate:     ${LR}"
echo " Logger:            ${LOGGER}"
echo " Cache Dir (HF):    ${HF_HOME}"
echo "=============================================================================="

# 3. Ensure Dataset Exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo ">> Dataset not found in ${DATA_DIR}. Running preprocessor first..."
    bash scripts/prepare_gsm8k_data.sh "${DATA_DIR}"
fi

# 4. Launch Distributed Training via torchrun (FSDP)
torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    optim.lr="${LR}" \
    engine=fsdp \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.total_epochs="${EPOCHS}" \
    "$@"

echo ""
echo "=============================================================================="
echo " Training Completed Successfully!"
echo " Checkpoint saved to: ${SAVE_PATH}"
echo "=============================================================================="
