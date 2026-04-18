#!/usr/bin/env bash
set -euo pipefail

bash scripts/01_prepare_and_process.sh
bash scripts/02_train_sft.sh
bash scripts/03_train_dpo.sh
