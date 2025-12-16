#!/usr/bin/env bash
# Minimal-ish nanochat speedrun tuned for low resources
# Put this file in the root of the karpathy/nanochat repo and run:
#
# You can override a few things when calling it, e.g.:
#   NPROC_PER_NODE=1 MODEL_DEPTH=8 DEVICE_BATCH_SIZE=4 bash speedrun_2x3090.sh

set -euo pipefail

########################
# User-tunable knobs  ##
########################

# How many GPUs to use (e.g. defaults to 2 for 2x3090)
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# Model size, default d12 - much smaller model than the official d20 speedrun
MODEL_DEPTH="${MODEL_DEPTH:-12}"

# Per-GPU batch size (smaller than default 32 to fit VRAM)
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

# DEVICE_BATCH_SIZE * CONTEXT_SIZE * NPROC_PER_NODE * X, should be around 524288
# X = 8
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
CONTEXT_SIZE="${CONTEXT_SIZE:-2048}"

# SFT
DEVICE_BATCH_SIZE_SFT="${DEVICE_BATCH_SIZE_SFT:-4}"
TOTAL_BATCH_SIZE_SFT="${TOTAL_BATCH_SIZE_SFT:-32}"

# How many FineWeb-EDU shards to download for pretraining.
# Original speedrun uses 240; here we default lower to reduce disk + download.
NUM_PRETRAIN_SHARDS="${NUM_PRETRAIN_SHARDS:-75}"

MODEL_TAG="${MODEL_TAG:-d12}"

DO_TOKENIZER="${DO_TOKENIZER:-1}"
DO_PRETREINING="${DO_PRETREINING:-1}"
DO_MIDTRAINING="${DO_MIDTRAINING:-1}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"

# Weights & Biases run name; set WANDB_RUN=yourname to enable logging.
export WANDB_RUN="nanochat-${MODEL_TAG}"

# Keep CPU threading modest so it doesn’t fight with the GPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "===== nanochat speedrun for 2x3090 ====="
echo "NPROC_PER_NODE      = ${NPROC_PER_NODE}"
echo "MODEL_DEPTH         = ${MODEL_DEPTH}"
echo "DEVICE_BATCH_SIZE   = ${DEVICE_BATCH_SIZE}"
echo "NUM_PRETRAIN_SHARDS = ${NUM_PRETRAIN_SHARDS}"
echo "NANOCHAT_BASE_DIR   = ${NANOCHAT_BASE_DIR}"
echo "NANOCHAT_BASE_DIR_STATIC  = ${NANOCHAT_BASE_DIR_STATIC}"
echo "WANDB_RUN           = ${WANDB_RUN}"

#################################
# Stage 0: Environment setup   ##
#################################

echo
echo ">>> [Stage 0] Python env + deps (uv, .venv) ..."
# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
#source $HOME/.local/bin/env ??

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv

# install the repo dependencies (GPU extras)
uv sync --extra gpu

# activate venv so that `python` uses the project's venv instead of system python
# shellcheck source=/dev/null
source .venv/bin/activate

#################################
# Stage 1: Tokenizer + data    ##
#################################

echo
echo ">>> [Stage 1] Install Rust toolchain + build rustbpe tokenizer ..."
# Install Rust / Cargo (non-interactive)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# shellcheck source=/dev/null
source "$HOME/.cargo/env"

if [[ DO_PRETREINING -eq 1 ]]; then

  if [[ DO_TOKENIZER -eq 1 && RESUME_FROM_STEP -eq -1 ]]; then
    # Initialize/reset report system
    echo
    echo ">>> nanochat.report reset ..."
    python -m nanochat.report reset

    echo
    echo ">>> Download a few shards first (for tokenizer) ..."
    python -m nanochat.dataset -n 8

    echo
    echo ">>> Kick off background download of remaining shards (for pretraining) ..."
    python -m nanochat.dataset -n "${NUM_PRETRAIN_SHARDS}" &
    DATASET_DOWNLOAD_PID=$!

    echo
    echo ">>> Train tokenizer on ~2B chars, then evaluate it ..."
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval

    #################################
    # Stage 2: Base pretraining    ##
    #################################

    echo
    echo ">>> Wait for background dataset download to finish ..."
    if ! wait "${DATASET_DOWNLOAD_PID}"; then
      echo "WARNING: dataset download process exited with non-zero status."
      echo "Training will still run but may loop over fewer shards."
    fi
  fi

  if [[ DO_MIDTRAINING -eq 1 ]]; then
    curl -L -o "${NANOCHAT_BASE_DIR}/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
  fi

  echo
  echo ">>> [Stage 2] Base pretraining (small d=${MODEL_DEPTH} model) ..."
  time torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.base_train -- \
      --depth="${MODEL_DEPTH}" \
      --device_batch_size="${DEVICE_BATCH_SIZE}" \
      --total_batch_size="${TOTAL_BATCH_SIZE}" \
      --max_seq_len="${CONTEXT_SIZE}" \
      --save_every=4500 \
      --core_metric_every=500 \
      --target_param_data_ratio=20 \
      --eval_every=500 \
      --model_tag="${MODEL_TAG}" \
      --resume_from_step="${RESUME_FROM_STEP}" \
      --sample_every=500 \
      --run="${WANDB_RUN}"

  sleep 10

  echo
  echo ">>> Base evals: base_loss ..."

  ((SPLIT_TOKENS=20*TOTAL_BATCH_SIZE))
  ((EVALS_DEVICE_BATCH_SIZE=DEVICE_BATCH_SIZE/2))

  time torchrun --standalone \
       --nproc_per_node="${NPROC_PER_NODE}" \
       -m scripts.base_loss -- \
         --device_batch_size="${EVALS_DEVICE_BATCH_SIZE}" \
         --split_tokens="${SPLIT_TOKENS}" \
         --model_tag="${MODEL_TAG}"

  echo
  echo ">>> Base evals: base_eval ..."
  time torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m scripts.base_eval --model-tag="${MODEL_TAG}"
fi

#################################
# Stage 3: Midtraining         ##
#################################

if [[ DO_MIDTRAINING -eq 1 ]]; then
  echo
  echo ">>> [Stage 3] Midtraining (SmolTalk + MMLU aux + GSM8K + identity) ..."
  time torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.mid_train -- \
      --device_batch_size="${DEVICE_BATCH_SIZE}" \
      --total_batch_size="${TOTAL_BATCH_SIZE}" \
      --max_seq_len="${CONTEXT_SIZE}" \
      --model_tag="${MODEL_TAG}" \
      --run="${WANDB_RUN}"

  echo
  echo ">>> Midtrained chat eval ..."
  time torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.chat_eval -- -i mid --model-tag="${MODEL_TAG}"
fi

#################################
# Stage 4: SFT                 ##
#################################

echo
echo ">>> [Stage 4] Supervised fine-tuning (SFT) ..."
time torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m scripts.chat_sft -- \
    --eval_every=50 \
    --device_batch_size="${DEVICE_BATCH_SIZE_SFT}" \
    --target_examples_per_step="${TOTAL_BATCH_SIZE_SFT}" \
    --run="${WANDB_RUN}" \
    --model_tag="${MODEL_TAG}"

echo
echo ">>> SFT chat eval ..."
time torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m scripts.chat_eval -- -i sft --model-tag="${MODEL_TAG}"

#################################
# Stage 5: Optional RL         ##
#################################
# RL is left commented out by default.
# Uncomment if you want to play with GSM8K RL finetuning later.

#: <<'RL_BLOCK'
# echo
# echo ">>> [Stage 5] RL on GSM8K (optional) ..."
# time torchrun \
#   --standalone \
#   --nproc_per_node="${NPROC_PER_NODE}" \
#   -m scripts.chat_rl -- \
#     --device_batch_size="${DEVICE_BATCH_SIZE}" \
#     --run="${WANDB_RUN}"
#
# echo
# echo ">>> RL eval on GSM8K ..."
# time torchrun \
#   --standalone \
#   --nproc_per_node="${NPROC_PER_NODE}" \
#   -m scripts.chat_eval -- -i rl -a GSM8K
#RL_BLOCK

#################################
# Stage 6: Report              ##
#################################

echo
echo ">>> [Stage 6] Generate final report.md ..."
python -m nanochat.report generate || echo "nanochat.report exited non-zero; check logs."

REPORT_DIR="${NANOCHAT_BASE_DIR}/report"
if [ -f "${REPORT_DIR}/report.md" ]; then
  # name it so you can accumulate multiple runs
  OUT_NAME="report_d${MODEL_DEPTH}${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).md"
  cp "${REPORT_DIR}/report.md" "./${OUT_NAME}"
  echo "Report copied to ./${OUT_NAME}"
else
  echo "WARNING: report.md not found in ${REPORT_DIR}; skip copy."
fi

echo
echo "===== Done. You can now run ====="
echo "  python -m scripts.chat_cli"
echo "or:"
echo "  python -m scripts.chat_web"
echo "================================="

python -m scripts.chat_web --model-tag="${MODEL_TAG}"
