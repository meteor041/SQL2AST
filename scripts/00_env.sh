#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=/home/pkuccadm/huwenp/emb/lxy/sql_rm
export DATA_ROOT=/data/huwenp/emb/data/ches
export TRAIN_JSON=${DATA_ROOT}/train.json
export DB_ROOT=${DATA_ROOT}/train_databases
export CANDIDATE_ROOT=${DATA_ROOT}/candidates/16_qwen7B_train

export EVAL_ROOT=${PROJECT_ROOT}/eval_results
export AST_ROOT=${PROJECT_ROOT}/eval_results_ast
export REPORT_ROOT=${PROJECT_ROOT}/reports
export DPO_DATA_ROOT=${PROJECT_ROOT}/data
export SFT_OUTPUT=${PROJECT_ROOT}/outputs/sft
export DPO_OUTPUT=${PROJECT_ROOT}/outputs/dpo

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "${EVAL_ROOT}" "${AST_ROOT}" "${REPORT_ROOT}" "${DPO_DATA_ROOT}" "${SFT_OUTPUT}" "${DPO_OUTPUT}"

cd "${PROJECT_ROOT}"
