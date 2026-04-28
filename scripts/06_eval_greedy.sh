#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/00_env.sh"
source "$(dirname "$0")/eval_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/06_eval_greedy.sh
  bash scripts/06_eval_greedy.sh sft
  bash scripts/06_eval_greedy.sh dpo
  bash scripts/06_eval_greedy.sh /path/to/model

Behavior:
  - Default model stage is `dpo`
  - `sft` uses ${SFT_OUTPUT}
  - `dpo` uses ${DPO_OUTPUT}
  - Passing a path uses that path as MODEL_SQL_GENERATE

Useful env vars:
  DATA_ROOT=/workspace/data/ches
  CSC_SQL_ROOT=/workspace/emb/lxy/csc_sql
  CUDA_VISIBLE_DEVICES=0
  TENSOR_PARALLEL_SIZE=1
  PROMPT_NAME=direct
  SYSTEM_PROMPT=none
  RUN_TIME=20260428_120000
  WAIT_FOR_GPU_IDLE=false
EOF
}

CSC_SQL_ROOT="${CSC_SQL_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/csc_sql}"

model_stage="${MODEL_STAGE:-dpo}"
explicit_model_path=""

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    sft|dpo)
      model_stage="$1"
      shift
      ;;
    *)
      explicit_model_path="$1"
      shift
      ;;
  esac
fi

if [[ -n "${explicit_model_path}" ]]; then
  model_path="${explicit_model_path}"
else
  case "${model_stage}" in
    sft)
      model_path="${SFT_OUTPUT}"
      ;;
    dpo)
      model_path="${DPO_OUTPUT}"
      ;;
    *)
      echo "Unsupported MODEL_STAGE=${model_stage}. Use sft, dpo, or pass a model path." >&2
      exit 1
      ;;
  esac
fi

if [[ ! -e "${model_path}" ]]; then
  echo "MODEL_SQL_GENERATE not found: ${model_path}" >&2
  exit 1
fi

stage_label="${model_stage}"
if [[ -n "${explicit_model_path}" ]]; then
  stage_label="custom"
fi

RUN_TIME="${RUN_TIME:-${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}_${stage_label}_greedy_n1}"
EVAL_LOG_DIR="${GREEDY_EVAL_LOG_DIR:-${REPORT_ROOT}/logs}"
mkdir -p "${EVAL_LOG_DIR}"
EVAL_LOG_FILE="${EVAL_LOG_DIR}/eval_greedy_${RUN_TIME}.log"
exec > >(tee -a "${EVAL_LOG_FILE}") 2>&1

echo "Greedy eval log file: ${EVAL_LOG_FILE}"
echo "RUN_TIME=${RUN_TIME}"

export EVAL_STEP="sql_generate"
export EVAL_MODE="greedy_search"
export PROMPT_NAME="${PROMPT_NAME:-direct}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-none}"
export N_SQL_GENERATE="${N_SQL_GENERATE:-1}"
export TEMPERATURE_SQL_GENERATE="${TEMPERATURE_SQL_GENERATE:-0.0}"
export EVAL_OUTPUT_DIR="${CSC_SQL_ROOT}/outputs/${RUN_TIME}"
export EVAL_OUTPUT_PREFIX="$(eval_result_prefix)"
export EVAL_MODEL_PATH="${model_path}"
export EVAL_WRAPPER_LOG_FILE="${EVAL_LOG_FILE}"
export EVAL_PIPELINE_LOG_FILE="${CSC_SQL_ROOT}/logs/run_pipeline_${RUN_TIME}.log"
export EVAL_WANDB_STAGE_LABEL="${stage_label}-greedy"
export EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-sql_rm-eval-${stage_label}-greedy-${RUN_TIME}}"

print_eval_wandb_status

trap 'eval_exit_code=$?; trap - EXIT; finalize_eval_wandb "${eval_exit_code}"; exit "${eval_exit_code}"' EXIT

prepare_cscsql_eval_data
wait_all_gpu_idle

cd "${CSC_SQL_ROOT}"
export PYTHONPATH="${CSC_SQL_ROOT}/src:${PYTHONPATH:-}"
configure_vllm_kernel_env
verify_vllm_runtime

echo "Running greedy search with MODEL_SQL_GENERATE=${model_path}"
echo "EVAL_MODE=${EVAL_MODE}"
echo "N_SQL_GENERATE=${N_SQL_GENERATE}"
echo "TEMPERATURE_SQL_GENERATE=${TEMPERATURE_SQL_GENERATE}"

MODEL_SQL_GENERATE="${model_path}" \
RUN_TIME="${RUN_TIME}" \
bash bin/process/run_sql_generate_eval.sh "$@"

echo "Expected csc_sql pipeline log: ${CSC_SQL_ROOT}/logs/run_pipeline_${RUN_TIME}.log"
echo "Expected greedy output dir: ${CSC_SQL_ROOT}/outputs/${RUN_TIME}"
