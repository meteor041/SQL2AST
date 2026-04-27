#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_env.sh"

DB_ENGINE="${DB_ENGINE:-sqlite}"
NUM_CPUS="${NUM_CPUS:-64}"
PROMPT_TEMPLATE_PATH="${PROJECT_ROOT}/src/template/prompt.txt"

validate_prompt_template() {
  local template_path="$1"
  local required_placeholders=("{db_engine}" "{db_details}" "{question}")

  if [[ ! -f "${template_path}" ]]; then
    echo "Prompt template not found: ${template_path}" >&2
    exit 1
  fi

  for placeholder in "${required_placeholders[@]}"; do
    if ! grep -q "${placeholder}" "${template_path}"; then
      echo "Missing placeholder ${placeholder} in ${template_path}" >&2
      exit 1
    fi
  done
}

prepare_location_file() {
  cat > .location <<EOF
TRAIN_DATA_PATH=${TRAIN_JSON}
TRAIN_DATABASE_PATH=${DB_ROOT}
SQL_PATH=${CANDIDATE_ROOT}
EVAL_OUTPUT_PATH=${EVAL_ROOT}
DB_ENGINE=${DB_ENGINE}
EOF
}

run_pipeline() {
  python eval.py --output-dir "${EVAL_ROOT}" --num-cpus "${NUM_CPUS}"

  python sql_to_ast.py "${EVAL_ROOT}" \
    -o "${AST_ROOT}" \
    --pattern '*_eval.json' \
    --dialect "${DB_ENGINE}"

  python src/calibrate.py \
    --eval-dir "${EVAL_ROOT}" \
    --train-data "${TRAIN_JSON}" \
    --database-root "${DB_ROOT}" \
    --output "${REPORT_ROOT}/distance_calibration.json"

  python src/build_pairs.py \
    --eval-dir "${EVAL_ROOT}" \
    --train-data "${TRAIN_JSON}" \
    --database-root "${DB_ROOT}" \
    --dialect "${DB_ENGINE}" \
    --output "${DPO_DATA_ROOT}/dpo_pairs.jsonl"
}

python -m pip install --upgrade pip
pip install -r requirements.txt
# Install torch for your CUDA version if it is not already available.
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

validate_prompt_template "${PROMPT_TEMPLATE_PATH}"
prepare_location_file
run_pipeline
