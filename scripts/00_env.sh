#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-/data/huwenp/emb/data/ches}"
export TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train.json}"
export DB_ROOT="${DB_ROOT:-${DATA_ROOT}/train_databases}"
export CANDIDATE_ROOT="${CANDIDATE_ROOT:-${DATA_ROOT}/candidates/16_qwen7B_train}"

export EVAL_ROOT="${EVAL_ROOT:-${DATA_ROOT}/eval_results}"
export AST_ROOT="${AST_ROOT:-${DATA_ROOT}/eval_results_ast}"
export REPORT_ROOT="${REPORT_ROOT:-${DATA_ROOT}/reports}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${DATA_ROOT}/data}"
export SFT_OUTPUT="${SFT_OUTPUT:-${DATA_ROOT}/outputs/sft}"
export DPO_OUTPUT="${DPO_OUTPUT:-${DATA_ROOT}/outputs/dpo}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected_devices="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}' | paste -sd, - || true)"
    export CUDA_VISIBLE_DEVICES="${detected_devices:-0}"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

mkdir -p "${EVAL_ROOT}" "${AST_ROOT}" "${REPORT_ROOT}" "${DPO_DATA_ROOT}" "${SFT_OUTPUT}" "${DPO_OUTPUT}"

cd "${PROJECT_ROOT}"
