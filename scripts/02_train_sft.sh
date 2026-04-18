#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

cat > configs/sft.local.yaml <<EOF
model_name_or_path: "Qwen/Qwen2.5-Coder-7B-Instruct"
train_data_path: "${TRAIN_JSON}"
database_root: "${DB_ROOT}"
output_dir: "${SFT_OUTPUT}"

num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
warmup_ratio: 0.05
lr_scheduler_type: "cosine"
max_seq_length: 2048

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
dataloader_num_workers: 4
EOF

torchrun --nproc_per_node=8 src/train_sft.py --config configs/sft.local.yaml
