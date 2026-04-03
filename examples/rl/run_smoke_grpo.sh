#!/usr/bin/env bash
set -euo pipefail

: "${TOKENIZER_MODEL:=}"
: "${TOKENIZER_TYPE:=HuggingFaceTokenizer}"
: "${GPUS:=2}"

if [[ -z "${TOKENIZER_MODEL}" ]]; then
  echo "Set TOKENIZER_MODEL to a local path or Hugging Face tokenizer identifier." >&2
  exit 1
fi

torchrun --nproc-per-node="${GPUS}" train_rl.py \
  --perform-rl-step \
  --use-mcore-models \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 \
  --global-batch-size 2 \
  --grpo-prompts-per-step 1 \
  --grpo-group-size 2 \
  --grpo-iterations 1 \
  --train-iters 1 \
  --seq-length 128 \
  --inference-max-seq-length 128 \
  --inference-max-requests 2 \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 512 \
  --num-attention-heads 4 \
  --max-position-embeddings 128 \
  --lr 1e-4 \
  --min-lr 1e-4 \
  --lr-decay-style constant \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --tokenizer-type "${TOKENIZER_TYPE}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --langrl-env-config examples/rl/environment_configs/smoke.yaml
