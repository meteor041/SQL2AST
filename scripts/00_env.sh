#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-/data/huwenp/emb/data/ches}"
export TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train.json}"
export DB_ROOT="${DB_ROOT:-${DATA_ROOT}/train_databases}"
export CANDIDATE_ROOT="${CANDIDATE_ROOT:-${DATA_ROOT}/candidates/16_qwen7B_train}"

REFRESH_DATA_ROOT="${REFRESH_DATA_ROOT:-${DATA_ROOT}/20260428_refresh}"
if [[ -d "/workspace/data/ches/data/20260428_refresh" ]]; then
  REFRESH_DATA_ROOT="/workspace/data/ches/data/20260428_refresh"
fi
export REFRESH_DATA_ROOT

default_sft_train_data_path="${TRAIN_JSON}"
if [[ -f "${REFRESH_DATA_ROOT}/sft_augmented.recommended.json" ]]; then
  default_sft_train_data_path="${REFRESH_DATA_ROOT}/sft_augmented.recommended.json"
elif [[ -f "/workspace/tmp/sft_augmented.recommended.json" ]]; then
  default_sft_train_data_path="/workspace/tmp/sft_augmented.recommended.json"
fi
export SFT_TRAIN_DATA_PATH="${SFT_TRAIN_DATA_PATH:-${default_sft_train_data_path}}"

default_dpo_pairs_path="${DPO_PAIRS_PATH:-}"
if [[ -z "${default_dpo_pairs_path}" ]]; then
  if [[ -f "${REFRESH_DATA_ROOT}/dpo_pairs.full.strict.jsonl" ]]; then
    default_dpo_pairs_path="${REFRESH_DATA_ROOT}/dpo_pairs.full.strict.jsonl"
  elif [[ -f "${DATA_ROOT}/data/dpo_pairs.jsonl" ]]; then
    default_dpo_pairs_path="${DATA_ROOT}/data/dpo_pairs.jsonl"
  fi
fi
export DPO_PAIRS_PATH="${DPO_PAIRS_PATH:-${default_dpo_pairs_path}}"

export EVAL_ROOT="${EVAL_ROOT:-${DATA_ROOT}/eval_results}"
export AST_ROOT="${AST_ROOT:-${DATA_ROOT}/eval_results_ast}"
export REPORT_ROOT="${REPORT_ROOT:-${DATA_ROOT}/reports}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${DATA_ROOT}/data}"
export SFT_OUTPUT="${SFT_OUTPUT:-${DATA_ROOT}/outputs/sft}"
export DPO_OUTPUT="${DPO_OUTPUT:-${DATA_ROOT}/outputs/dpo}"

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${PROJECT_ROOT}/configs/wandb.local.env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-wandb}"
else
  export TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-sql_rm}"
export WANDB_DIR="${WANDB_DIR:-${DATA_ROOT}/outputs/wandb}"

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

mkdir -p "${EVAL_ROOT}" "${AST_ROOT}" "${REPORT_ROOT}" "${DPO_DATA_ROOT}" "${SFT_OUTPUT}" "${DPO_OUTPUT}" "${WANDB_DIR}"

cd "${PROJECT_ROOT}"
