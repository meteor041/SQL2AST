#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/00_env.sh"
source "$(dirname "$0")/eval_common.sh"

CSC_SQL_ROOT="${CSC_SQL_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)/csc_sql}"

prepare_cscsql_eval_data
wait_all_gpu_idle

cd "${CSC_SQL_ROOT}"
export PYTHONPATH="${CSC_SQL_ROOT}/src:${PYTHONPATH:-}"

MODEL_SQL_GENERATE="${DPO_OUTPUT}" \
PROMPT_NAME="${PROMPT_NAME:-direct}" \
SYSTEM_PROMPT="${SYSTEM_PROMPT:-none}" \
N_SQL_GENERATE="${N_SQL_GENERATE:-8}" \
bash bin/process/run_sql_generate_eval.sh "$@"
