#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyyaml datasets transformers peft trl accelerate sentencepiece scipy scikit-learn
# Install torch for your CUDA version if it is not already available.
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cat > .location <<EOF
TRAIN_DATA_PATH=${TRAIN_JSON}
TRAIN_DATABASE_PATH=${DB_ROOT}
SQL_PATH=${CANDIDATE_ROOT}
EVAL_OUTPUT_PATH=${EVAL_ROOT}
EOF

python eval.py --output-dir "${EVAL_ROOT}" --num-cpus 64

python sql_to_ast.py "${EVAL_ROOT}" \
  -o "${AST_ROOT}" \
  --pattern '*_eval.json' \
  --dialect sqlite

python src/calibrate.py \
  --eval-dir "${EVAL_ROOT}" \
  --train-data "${TRAIN_JSON}" \
  --database-root "${DB_ROOT}" \
  --output "${REPORT_ROOT}/distance_calibration.json"

python src/build_pairs.py \
  --eval-dir "${EVAL_ROOT}" \
  --train-data "${TRAIN_JSON}" \
  --database-root "${DB_ROOT}" \
  --output "${DPO_DATA_ROOT}/dpo_pairs.jsonl"
