#!/usr/bin/env bash
# Prepares the GSM8K dataset for Supervised Fine-Tuning (SFT)
# Uses examples/data_preprocess/gsm8k_multiturn_sft.py which builds the required "messages" structure.

set -euo pipefail

# Default save path: /scratch/$USER/data/gsm8k_sft (falls back to ~/data/gsm8k_sft if /scratch is not available)
if [ -d "/scratch/$USER" ]; then
    DEFAULT_DATA_DIR="/scratch/$USER/data/gsm8k_sft"
else
    DEFAULT_DATA_DIR="$HOME/data/gsm8k_sft"
fi

DATA_DIR="${1:-$DEFAULT_DATA_DIR}"

echo "=================================================="
echo " Preparing GSM8K Dataset for veRL SFT"
echo " Target Output Directory: ${DATA_DIR}"
echo "=================================================="

mkdir -p "${DATA_DIR}"

python3 examples/data_preprocess/gsm8k_multiturn_sft.py \
    --local_save_dir "${DATA_DIR}"

echo ""
echo " Successfully preprocessed GSM8K dataset!"
echo " Train parquet: ${DATA_DIR}/train.parquet"
echo " Test parquet:  ${DATA_DIR}/test.parquet"
echo "=================================================="
