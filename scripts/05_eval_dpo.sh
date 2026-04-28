#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/00_env.sh"
source "$(dirname "$0")/eval_common.sh"

CSC_SQL_ROOT="${CSC_SQL_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/csc_sql}"
RUN_TIME="${RUN_TIME:-${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}_dpo_sampling_n8_no_merge}"
EVAL_LOG_DIR="${DPO_EVAL_LOG_DIR:-${REPORT_ROOT}/logs}"
mkdir -p "${EVAL_LOG_DIR}"
EVAL_LOG_FILE="${EVAL_LOG_DIR}/eval_dpo_sampling_${RUN_TIME}.log"
exec > >(tee -a "${EVAL_LOG_FILE}") 2>&1

echo "DPO sampling eval log file: ${EVAL_LOG_FILE}"
echo "RUN_TIME=${RUN_TIME}"
echo "N_SQL_GENERATE=${N_SQL_GENERATE:-8}"
echo "TEMPERATURE_SQL_GENERATE=${TEMPERATURE_SQL_GENERATE:-0.8}"
echo "EVAL_STEP=sql_generate (no merge)"
echo "MODEL_SQL_MERGE=none"

export EVAL_STEP="sql_generate"
export EVAL_MODE="${EVAL_MODE:-major_voting}"
export PROMPT_NAME="${PROMPT_NAME:-direct}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-none}"
export N_SQL_GENERATE="${N_SQL_GENERATE:-8}"
export TEMPERATURE_SQL_GENERATE="${TEMPERATURE_SQL_GENERATE:-0.8}"
export EVAL_OUTPUT_DIR="${CSC_SQL_ROOT}/outputs/${RUN_TIME}"
export EVAL_OUTPUT_PREFIX="$(eval_result_prefix)"
export EVAL_MODEL_PATH="${DPO_OUTPUT}"
export EVAL_WRAPPER_LOG_FILE="${EVAL_LOG_FILE}"
export EVAL_PIPELINE_LOG_FILE="${CSC_SQL_ROOT}/logs/run_pipeline_${RUN_TIME}.log"
export EVAL_WANDB_STAGE_LABEL="dpo-sampling"
export EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-sql_rm-eval-dpo-sampling-${RUN_TIME}}"

print_eval_wandb_status

trap 'eval_exit_code=$?; trap - EXIT; finalize_eval_wandb "${eval_exit_code}"; exit "${eval_exit_code}"' EXIT

prepare_cscsql_eval_data
wait_all_gpu_idle

cd "${CSC_SQL_ROOT}"
export PYTHONPATH="${CSC_SQL_ROOT}/src:${PYTHONPATH:-}"
configure_vllm_kernel_env
verify_vllm_runtime

MODEL_SQL_GENERATE="${DPO_OUTPUT}" \
PROMPT_NAME="${PROMPT_NAME}" \
SYSTEM_PROMPT="${SYSTEM_PROMPT}" \
RUN_TIME="${RUN_TIME}" \
N_SQL_GENERATE="${N_SQL_GENERATE}" \
TEMPERATURE_SQL_GENERATE="${TEMPERATURE_SQL_GENERATE}" \
bash bin/process/run_sql_generate_eval.sh "$@"

echo "Expected csc_sql pipeline log: ${CSC_SQL_ROOT}/logs/run_pipeline_${RUN_TIME}.log"
echo "Expected sampling output dir: ${CSC_SQL_ROOT}/outputs/${RUN_TIME}"
