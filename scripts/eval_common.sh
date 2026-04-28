#!/usr/bin/env bash

json_has_rows() {
  local json_path="$1"

  [[ -f "${json_path}" ]] || return 1

  python3 -c '
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

sys.exit(0 if isinstance(data, list) and len(data) > 0 else 1)
' "${json_path}"
}

verify_vllm_runtime() {
  local python_bin="${PYTHON_BIN:-python3}"

  "${python_bin}" - <<'PY'
import importlib
import sys

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch before vLLM preflight: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    importlib.import_module("vllm._C")
except Exception as exc:
    print("vLLM runtime preflight failed.", file=sys.stderr)
    print(f"python={sys.executable}", file=sys.stderr)
    print(f"torch={torch.__version__}, torch.cuda={torch.version.cuda}", file=sys.stderr)
    print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)

    err_text = str(exc)
    if "libcudart.so.13" in err_text:
        print(
            "Detected a CUDA runtime mismatch: the installed vllm build expects CUDA 13, "
            "but this environment only exposes CUDA 12.x libraries.",
            file=sys.stderr,
        )
        print(
            "Reinstall a CUDA 12.8-compatible vllm build in the active virtualenv before rerunning eval.",
            file=sys.stderr,
        )
    raise SystemExit(1)
PY
}

configure_vllm_kernel_env() {
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
  export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
  export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

  echo "vLLM kernel settings:"
  echo "  VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
  echo "  VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM}"
  echo "  VLLM_DEEP_GEMM_WARMUP=${VLLM_DEEP_GEMM_WARMUP}"
}

wandb_eval_requested() {
  [[ "${EVAL_REPORT_TO:-${TRAIN_REPORT_TO:-none}}" == "wandb" ]]
}

print_eval_wandb_status() {
  local eval_report_to="${EVAL_REPORT_TO:-${TRAIN_REPORT_TO:-none}}"

  if [[ "${eval_report_to}" == "wandb" ]]; then
    echo "Eval wandb enabled: project=${EVAL_WANDB_PROJECT:-${WANDB_PROJECT}}"
    echo "Eval wandb dir: ${WANDB_DIR}"
    echo "Eval wandb base_url: ${WANDB_BASE_URL:-https://api.wandb.ai}"
  else
    echo "Eval wandb disabled; set EVAL_REPORT_TO=wandb (or TRAIN_REPORT_TO=wandb) to enable remote eval runs."
  fi
}

eval_result_prefix() {
  local eval_mode="${EVAL_MODE:-major_voting}"
  local prompt_name="${PROMPT_NAME:-think}"
  local eval_step="${EVAL_STEP:-sql_generate}"
  local prefix="sampling"

  if [[ "${eval_mode}" == "greedy_search" ]]; then
    prefix="greedy_search"
  fi

  printf "%s_%s_%s" "${prefix}" "${prompt_name}" "${eval_step}"
}

finalize_eval_wandb() {
  local exit_code="${1:-0}"

  if [[ "${EVAL_WANDB_FINALIZED:-0}" == "1" ]]; then
    return 0
  fi
  export EVAL_WANDB_FINALIZED=1

  if ! wandb_eval_requested; then
    return 0
  fi

  if [[ -z "${RUN_TIME:-}" ]]; then
    echo "Skip eval wandb logging because RUN_TIME is empty." >&2
    return 0
  fi

  local eval_output_dir="${EVAL_OUTPUT_DIR:-}"
  local eval_prefix="${EVAL_OUTPUT_PREFIX:-}"
  if [[ -z "${eval_prefix}" ]]; then
    eval_prefix="$(eval_result_prefix)"
  fi

  local eval_base_path="${eval_output_dir}/${eval_prefix}"
  local python_bin="${PYTHON_BIN:-python3}"
  local strict_flag=()
  if [[ "${EVAL_WANDB_STRICT:-0}" == "1" || "${EVAL_WANDB_STRICT:-false}" == "true" ]]; then
    strict_flag+=(--strict)
  fi

  "${python_bin}" "${PROJECT_ROOT}/src/log_eval_to_wandb.py" \
    --report-to "${EVAL_REPORT_TO:-${TRAIN_REPORT_TO:-none}}" \
    --run-name "${EVAL_WANDB_RUN_NAME:-sql_rm-eval-${EVAL_WANDB_STAGE_LABEL:-eval}-${RUN_TIME}}" \
    --project "${EVAL_WANDB_PROJECT:-${WANDB_PROJECT}}" \
    --group "${EVAL_WANDB_GROUP:-${WANDB_RUN_GROUP:-}}" \
    --job-type "${EVAL_WANDB_JOB_TYPE:-eval}" \
    --stage-label "${EVAL_WANDB_STAGE_LABEL:-eval}" \
    --run-time "${RUN_TIME}" \
    --dataset-mode "${DATASET_MODE:-dev}" \
    --prompt-name "${PROMPT_NAME:-think}" \
    --eval-mode "${EVAL_MODE:-major_voting}" \
    --eval-step "${EVAL_STEP:-sql_generate}" \
    --model-path "${EVAL_MODEL_PATH:-}" \
    --output-dir "${eval_output_dir}" \
    --metric-json "${EVAL_METRIC_JSON_PATH:-${eval_base_path}_metric.json}" \
    --predicted-sql "${EVAL_RESULT_SQL_PATH:-${eval_base_path}_pred_major_voting_sqls.sql}" \
    --raw-pred-json "${EVAL_RAW_PRED_JSON_PATH:-${eval_base_path}.json}" \
    --arg-json "${EVAL_ARG_JSON_PATH:-${eval_output_dir}/arg_${EVAL_STEP:-sql_generate}.json}" \
    --wrapper-log "${EVAL_WRAPPER_LOG_FILE:-}" \
    --pipeline-log "${EVAL_PIPELINE_LOG_FILE:-}" \
    --tag "eval" \
    --tag "${EVAL_WANDB_STAGE_LABEL:-eval}" \
    --tag "${DATASET_MODE:-dev}" \
    --tag "${EVAL_MODE:-major_voting}" \
    --exit-code "${exit_code}" \
    "${strict_flag[@]}"
}

wait_all_gpu_idle() {
  if [[ "${WAIT_FOR_GPU_IDLE:-true}" == "false" || "${WAIT_FOR_GPU_IDLE:-true}" == "0" ]]; then
    echo "WAIT_FOR_GPU_IDLE=${WAIT_FOR_GPU_IDLE}; skip waiting for idle GPUs."
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot wait for idle GPUs." >&2
    return 1
  fi

  local max_memory_mb="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
  local max_util="${GPU_IDLE_MAX_UTIL:-5}"
  local interval="${GPU_IDLE_CHECK_INTERVAL:-60}"
  local devices_csv="${CUDA_VISIBLE_DEVICES:-}"
  local devices=()

  if [[ -n "${devices_csv}" ]]; then
    IFS=',' read -r -a devices <<< "${devices_csv}"
  else
    mapfile -t devices < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  fi

  echo "Waiting for GPUs (${devices[*]}) to be idle: memory <= ${max_memory_mb} MiB, util <= ${max_util}%."

  while true; do
    local all_idle=1
    local status_lines=()

    for gpu in "${devices[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      if [[ -z "${gpu}" ]]; then
        continue
      fi

      local stats
      if ! stats="$(nvidia-smi --id="${gpu}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
        echo "Failed to query GPU '${gpu}' with nvidia-smi." >&2
        return 1
      fi

      local index memory_used util
      IFS=',' read -r index memory_used util <<< "${stats}"
      index="${index//[[:space:]]/}"
      memory_used="${memory_used//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      status_lines+=("gpu${index}: ${memory_used}MiB, ${util}%")

      if (( memory_used > max_memory_mb || util > max_util )); then
        all_idle=0
      fi
    done

    if (( all_idle == 1 )); then
      echo "All requested GPUs are idle: ${status_lines[*]}"
      return 0
    fi

    echo "GPUs busy: ${status_lines[*]}; recheck in ${interval}s."
    sleep "${interval}"
  done
}

prepare_cscsql_eval_data() {
  local dataset_mode="${DATASET_MODE:-dev}"
  local dataset_base_dir="${DATASET_BASE_DIR:-${DATA_ROOT}}"
  local datafile_path="${DATAFILE_PATH:-}"

  if [[ -z "${datafile_path}" ]]; then
    case "${dataset_mode}" in
      dev)
        datafile_path="${dataset_base_dir}/dev_bird.json"
        ;;
      test)
        datafile_path="${dataset_base_dir}/test_bird.json"
        ;;
      *)
        echo "Unsupported DATASET_MODE=${dataset_mode}. Use dev or test." >&2
        exit 1
        ;;
    esac
  fi

  if json_has_rows "${datafile_path}"; then
    export DATAFILE_PATH="${datafile_path}"
    return 0
  fi

  local raw_data_file
  local db_path
  local tables_path
  local save_index_path

  case "${dataset_mode}" in
    dev)
      raw_data_file="${dataset_base_dir}/dev.json"
      db_path="${dataset_base_dir}/dev_databases"
      tables_path="${dataset_base_dir}/dev_tables.json"
      save_index_path="${dataset_base_dir}/dev_db_contents_index"
      ;;
    test)
      raw_data_file="${dataset_base_dir}/test.json"
      db_path="${dataset_base_dir}/test_databases"
      tables_path="${dataset_base_dir}/test_tables.json"
      save_index_path="${dataset_base_dir}/test_db_contents_index"
      ;;
  esac

  for required_path in "${raw_data_file}" "${db_path}" "${tables_path}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required evaluation input not found: ${required_path}" >&2
      exit 1
    fi
  done

  echo "Prepared prompt file is missing or empty: ${datafile_path}"
  echo "Generating it from ${raw_data_file}"

  PYTHONPATH="${CSC_SQL_ROOT}/src:${PYTHONPATH:-}" python3 -m cscsql.service.process.process_dataset \
    --input_data_file "${raw_data_file}" \
    --output_data_file "${datafile_path}" \
    --db_path "${db_path}" \
    --tables "${tables_path}" \
    --source "${DATASET_NAME:-bird}" \
    --mode "${dataset_mode}" \
    --value_limit_num "${VALUE_LIMIT_NUM:-2}" \
    --db_content_index_path "${save_index_path}"

  if ! json_has_rows "${datafile_path}"; then
    echo "Generated prompt file is still empty: ${datafile_path}" >&2
    exit 1
  fi

  export DATAFILE_PATH="${datafile_path}"
}
