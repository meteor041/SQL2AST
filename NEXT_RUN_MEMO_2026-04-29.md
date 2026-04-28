SFT / DPO next-run memo

Data saved
- SFT merged training set:
  `/workspace/data/ches/data/20260428_refresh/sft_augmented.recommended.json`
- SFT augmented-only records:
  `/workspace/data/ches/data/20260428_refresh/sft_augmented.recommended.only.json`
- DPO strict training set:
  `/workspace/data/ches/data/20260428_refresh/dpo_pairs.full.strict.jsonl`
- DPO loose training set:
  `/workspace/data/ches/data/20260428_refresh/dpo_pairs.full.loose.jsonl`
- DPO full metadata file:
  `/workspace/data/ches/data/20260428_refresh/dpo_pairs.with_meta.full.jsonl`

Current sizes
- SFT original: 9428
- SFT augmented added: 4166
- SFT merged total: 13594
- DPO strict: 17247
- DPO loose: 44411
- DPO full with metadata: 338048

Current defaults already switched
- `scripts/00_env.sh` now defaults:
  - `SFT_TRAIN_DATA_PATH` -> `.../20260428_refresh/sft_augmented.recommended.json`
  - `DPO_PAIRS_PATH` -> `.../20260428_refresh/dpo_pairs.full.strict.jsonl`
- `configs/sft.local.yaml` points to the saved SFT augmented file.
- `configs/dpo.local.yaml` points to the saved DPO strict file.

Recommended run order
1. Run SFT on the augmented dataset.
2. Evaluate SFT if needed.
3. Run DPO on the strict dataset first.
4. Use the loose dataset only as a comparison run.

Direct commands
```bash
bash scripts/02_train_sft.sh
bash scripts/03_train_dpo.sh
```

If you want the loose DPO set instead
```bash
export DPO_PAIRS_PATH=/workspace/data/ches/data/20260428_refresh/dpo_pairs.full.loose.jsonl
bash scripts/03_train_dpo.sh
```

Notes
- `train_databases` contains both flat and nested sqlite layouts. `train_sft.py` and `build_pairs.py` were patched to handle both.
- `works_cycles.sqlite` was added and the missing DPO subset was backfilled.
- SFT augmentation rule used for the recommended file:
  - execution-correct sampled SQL
  - normalized SQL different from gold
  - max 1 extra SQL per original sample
  - AST distance to gold <= 0.2
- DPO strict rule:
  - only `correct_wrong`
  - chosen must be correct
  - rejected must be wrong
  - max 4 pairs per prompt
  - tokenizer length filters still use `max_prompt_tokens=768`, `max_total_tokens=1024`

Optional improvement before long DPO runs
- Consider increasing DPO lengths to:
  - `max_prompt_length: 1024`
  - `max_seq_length: 1536`
- If memory is tight, reduce `per_device_train_batch_size`.
