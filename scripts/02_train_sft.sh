#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
sft_report_to="${SFT_REPORT_TO:-${TRAIN_REPORT_TO:-none}}"
sft_run_name="${SFT_RUN_NAME:-sql_rm-sft-${RUN_TIMESTAMP}}"

cat > configs/sft.local.yaml <<EOF
model_name_or_path: "${MODEL_NAME_OR_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
train_data_path: "${SFT_TRAIN_DATA_PATH}"
database_root: "${DB_ROOT}"
output_dir: "${SFT_OUTPUT}"

num_train_epochs: 3
per_device_train_batch_size: ${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}
gradient_accumulation_steps: ${SFT_GRADIENT_ACCUMULATION_STEPS:-16}
learning_rate: 2.0e-5
warmup_ratio: 0.05
lr_scheduler_type: "cosine"
max_seq_length: ${SFT_MAX_SEQ_LENGTH:-1024}

use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

bf16: true
fp16: false
seed: 42
logging_steps: 10
save_steps: 200
eval_steps: 200
dataloader_num_workers: ${SFT_DATALOADER_NUM_WORKERS:-0}
report_to: "${sft_report_to}"
run_name: "${sft_run_name}"
EOF

LOG_DIR="${SFT_LOG_DIR:-${SFT_OUTPUT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sft_${RUN_TIMESTAMP}.log"
CONFIG_SNAPSHOT="${LOG_DIR}/sft_${RUN_TIMESTAMP}.yaml"
ENV_SNAPSHOT="${LOG_DIR}/sft_${RUN_TIMESTAMP}.env"

exec > >(tee -a "${LOG_FILE}") 2>&1

cp configs/sft.local.yaml "${CONFIG_SNAPSHOT}"
env | sort > "${ENV_SNAPSHOT}"

echo "SFT log file: ${LOG_FILE}"
echo "SFT config snapshot: ${CONFIG_SNAPSHOT}"
echo "SFT env snapshot: ${ENV_SNAPSHOT}"
echo "SFT train data path: ${SFT_TRAIN_DATA_PATH}"
echo "SFT run name: ${sft_run_name}"
echo "SFT report_to: ${sft_report_to}"
if [[ "${sft_report_to}" == "wandb" ]]; then
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "WANDB_DIR=${WANDB_DIR}"
  echo "WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}"
else
  echo "wandb disabled; set WANDB_API_KEY or TRAIN_REPORT_TO=wandb to enable remote loss curves."
fi

MASTER_PORT="${MASTER_PORT:-29501}"
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  nproc_per_node="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  nproc_per_node="${#visible_devices[@]}"
else
  nproc_per_node=8
fi

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

wait_all_gpu_idle

echo "Starting SFT training with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} and nproc_per_node=${nproc_per_node}"
torchrun --nproc_per_node="${nproc_per_node}" --master_port="${MASTER_PORT}" src/train_sft.py --config configs/sft.local.yaml "$@"
