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
