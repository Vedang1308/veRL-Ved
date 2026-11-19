# Step-by-Step Guide: Running GSM8K SFT on ASU Sol Supercomputer

This guide walks you through every step of fine-tuning `Qwen2.5-1.5B` on the GSM8K math dataset using `veRL` and testing the result.

---

## Overview: What We Are Doing

1. **Allocating GPU Compute**: Getting an interactive terminal on a GPU worker node with 2x A100 GPUs.
2. **Environment Setup**: Creating a Python 3.12 environment with `uv` and installing `veRL` dependencies.
3. **Data Preparation**: Downloading and converting the 8,000 GSM8K math problems into `.parquet` format.
4. **Supervised Fine-Tuning (SFT)**: Running distributed training across the 2 GPUs using PyTorch FSDP.
5. **Evaluation**: Running sample test questions through our newly trained model to verify it learned step-by-step reasoning.

---

## Step 1: Connect to Sol & Request 2x A100 GPUs

Open your local terminal and log in to Sol:
```bash
ssh <ASURITE>@sol.asu.edu
```

Once logged into the login node, request an interactive session with 2 A100 GPUs:
```bash
interactive -p gpu -q wildfire --gres=gpu:a100:2 -c 8 -t 02:00:00
```
> *(If `interactive` is not available, use the standard SLURM equivalent:)*
> ```bash
> salloc --partition=gpu --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=02:00:00
> ```

* **Layman's explanation:** You are asking the cluster manager: *"Please assign me 2 A100 graphics cards and 8 CPU cores for 2 hours so I can work in an active terminal."*
* When the prompt changes (e.g. to `[asurite@gpu-node-XX ~]$`), you are inside the GPU worker node!

---

## Step 2: Navigate to `/scratch` and Clone Your Repository

Always work in `/scratch/$USER` on Sol to avoid running out of disk space on your home folder quota:

```bash
cd /scratch/$USER
git clone https://github.com/Vedang1308/veRL-Ved.git
cd veRL-Ved
```

* **Layman's explanation:** Your home directory (`~`) only has a small storage limit. `/scratch` has hundreds of gigabytes of fast NVMe storage, which is where models and datasets must live.

---

## Step 3: Set Up Python and veRL Dependencies

Load the system CUDA module and use `uv` for fast dependency installation:

```bash
# 1. Load CUDA 12
module purge
module load cuda/12.4

# 2. Redirect Hugging Face caches to scratch
export HF_HOME=/scratch/$USER/hf_cache
export TMPDIR=/scratch/$USER/tmp
mkdir -p $HF_HOME $TMPDIR

# 3. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 4. Create virtual environment with Python 3.12 and install FSDP dependencies
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra fsdp
```

* **Layman's explanation:** This creates an isolated software sandbox containing PyTorch, CUDA libraries, and veRL without interfering with any other software on the supercomputer.

---

## Step 4: Prepare the GSM8K Math Dataset

Run the data preparation script:

```bash
bash scripts/prepare_gsm8k_data.sh
```

* **Layman's explanation:** This script contacts Hugging Face, downloads the 8,000 grade-school math word problems, and formats each problem into a conversation:
  - **User:** The math question + *"Let's think step by step and output the final answer after ####"*
  - **Assistant:** The full worked-out step-by-step reasoning + the final answer.
* Files saved to: `/scratch/$USER/data/gsm8k_sft/train.parquet` and `test.parquet`.

---

## Step 5: Launch Supervised Fine-Tuning (SFT)

Run the SFT training script:

```bash
bash scripts/run_gsm8k_sft.sh 2 Qwen/Qwen2.5-1.5B
```

* **Arguments explained:**
  - `2`: Use 2 GPUs.
  - `Qwen/Qwen2.5-1.5B`: The base AI model to train.
* **What happens during training:**
  - The script splits the model across both A100 GPUs using PyTorch FSDP (Fully Sharded Data Parallel).
  - It shows a progress bar with training loss (`loss=...`). You will watch the loss steadily drop from ~1.5+ down towards ~0.1–0.3.
  - At the end of training (about 15–25 minutes), it automatically saves the checkpoint and exports full Hugging Face format weights into `/scratch/$USER/checkpoints/qwen2.5-1.5b-gsm8k-sft/global_step_.../huggingface`.

---

## Step 6: Test the Trained Model & Show Proof

To test your newly trained model on real math problems, run:

```bash
python3 scripts/test_gsm8k.py \
    --model_path /scratch/$USER/checkpoints/qwen2.5-1.5b-gsm8k-sft/global_step_*/huggingface
```

* **Layman's explanation:** This script feeds real math questions to your newly trained model, prints the step-by-step thinking the model generated, extracts the final number after `####`, and checks if it got the question right!
* Copy-paste this terminal output — this is your direct proof for the professor!
