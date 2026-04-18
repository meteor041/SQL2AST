#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

cat > configs/dpo.local.yaml <<EOF
model_name_or_path: "${SFT_OUTPUT}"
ref_model_name_or_path: null
dpo_pairs_path: "${DPO_DATA_ROOT}/dpo_pairs.jsonl"
output_dir: "${DPO_OUTPUT}"

beta: 0.1
alpha: 1.0

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
warmup_ratio: 0.1
lr_scheduler_type: "cosine"
max_seq_length: 2048
max_prompt_length: 1536
eval_split: 0.05

use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

bf16: true
fp16: false
seed: 42
logging_steps: 10
save_steps: 100
dataloader_num_workers: 2
EOF

torchrun --nproc_per_node=8 src/train_dpo.py --config configs/dpo.local.yaml
