"""SFT warm-up training for NL2SQL.

Loads BIRD train.json, formats (prompt, SQL) pairs, and fine-tunes the base
model with LoRA using TRL's SFTTrainer.

Usage
-----
python src/train_sft.py --config configs/sft.yaml
python src/train_sft.py --config configs/sft.yaml --learning_rate 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: PyYAML.  Run `pip install pyyaml`.")


@dataclass
class SFTConfig:
    # ── model ──────────────────────────────────────────────────────────────
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    attn_implementation: str | None = None
    # ── data ───────────────────────────────────────────────────────────────
    train_data_path: str = ""       # path to BIRD train.json
    database_root:   str = ""       # path to SQLite databases root
    output_dir:      str = "outputs/sft"
    # ── training hyper-parameters ──────────────────────────────────────────
    num_train_epochs:            int   = 3
    per_device_train_batch_size: int   = 4
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 2e-5
    warmup_ratio:                float = 0.05
    lr_scheduler_type:           str   = "cosine"
    max_seq_length:              int   = 2048
    # ── LoRA ───────────────────────────────────────────────────────────────
    use_lora:             bool      = True
    lora_r:               int       = 16
    lora_alpha:           int       = 32
    lora_dropout:         float     = 0.05
    lora_target_modules:  list[str] | None = None
    # ── misc ───────────────────────────────────────────────────────────────
    seed:                  int  = 42
    bf16:                  bool = True
    fp16:                  bool = False
    logging_steps:         int  = 10
    save_steps:            int  = 200
    eval_steps:            int  = 200
    eval_ratio:            float = 0.0
    dataloader_num_workers: int  = 4
    report_to:             str | list[str] | None = "none"
    run_name:              str | None = None


def load_config(config_path: Path, overrides: dict[str, Any] | None = None) -> SFTConfig:
    """Load SFTConfig from YAML and apply CLI overrides."""
    with config_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        data.update(overrides)

    valid = {f.name for f in fields(SFTConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    return SFTConfig(**{k: v for k, v in data.items() if k in valid})


def build_sft_dataset(
    train_data: list[dict],
    database_root: Path,
    tokenizer: Any,
    max_seq_length: int = 2048,
    log_every: int = 200,
) -> Any:
    """Build a HuggingFace Dataset of tokenized (prompt + SQL) sequences.

    Each sample's labels mask the prompt tokens so that loss is computed
    only on the SQL completion.
    """
    from datasets import Dataset

    from src.utils.cscsql_prompt import (
        CSCSQLPromptUnavailableError,
        build_cscsql_prompt,
    )
    from src.utils.prompt import format_sql_response

    def _resolve_db(db_id: str) -> Path:
        nested = database_root / db_id / f"{db_id}.sqlite"
        if nested.exists():
            return nested
        flat = database_root / f"{db_id}.sqlite"
        return flat

    records: list[dict[str, list[int]]] = []
    total = len(train_data)

    # In DDP, all ranks execute preprocessing. Keep logs on rank 0 to avoid noise.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    should_log = local_rank == 0
    started = time.time()
    kept = 0

    if should_log:
        print(f"[SFT] Preprocessing started: total_samples={total}")

    for idx, item in enumerate(train_data, start=1):
        question = item.get("question", "")
        gold_sql = item.get("SQL", "")
        db_id    = item.get("db_id", "")
        evidence = item.get("evidence", "")

        if not question or not gold_sql or not db_id:
            if should_log and (idx % log_every == 0 or idx == total):
                elapsed = time.time() - started
                print(
                    f"[SFT] Preprocessing {idx}/{total} "
                    f"({idx / max(total, 1) * 100:.1f}%) kept={kept} elapsed={elapsed:.1f}s"
                )
            continue

        try:
            prompt = build_cscsql_prompt(
                question=question,
                db_id=db_id,
                database_root=database_root,
                evidence=evidence,
                gold_sql=gold_sql,
                mode="eval",
            )
        except CSCSQLPromptUnavailableError:
            raise
        except Exception:
            if should_log and (idx % log_every == 0 or idx == total):
                elapsed = time.time() - started
                print(
                    f"[SFT] Preprocessing {idx}/{total} "
                    f"({idx / max(total, 1) * 100:.1f}%) kept={kept} elapsed={elapsed:.1f}s"
                )
            continue

        completion  = format_sql_response(gold_sql)
        full_text   = prompt + completion

        prompt_ids     = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids       = tokenizer.encode(full_text, add_special_tokens=True,
                                          truncation=True, max_length=max_seq_length)

        # Mask prompt tokens in labels (−100 means "ignore in loss")
        n_prompt = min(len(prompt_ids), len(full_ids) - 1)
        labels   = [-100] * n_prompt + full_ids[n_prompt:]

        records.append({
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        })
        kept += 1

        if should_log and (idx % log_every == 0 or idx == total):
            elapsed = time.time() - started
            print(
                f"[SFT] Preprocessing {idx}/{total} "
                f"({idx / max(total, 1) * 100:.1f}%) kept={kept} elapsed={elapsed:.1f}s"
            )

    if should_log:
        elapsed = time.time() - started
        print(
            f"[SFT] Preprocessing finished: kept={kept}/{total}, "
            f"elapsed={elapsed:.1f}s"
        )

    return Dataset.from_list(records)


def split_train_eval_data(
    train_data: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic train/eval holdout from raw train.json records."""
    if eval_ratio <= 0:
        return train_data, []
    if eval_ratio >= 1:
        raise ValueError(f"eval_ratio must be in [0, 1). Got: {eval_ratio}")
    if len(train_data) < 2:
        return train_data, []

    eval_count = max(1, math.ceil(len(train_data) * eval_ratio))
    eval_count = min(eval_count, len(train_data) - 1)

    rng = random.Random(seed)
    indices = list(range(len(train_data)))
    rng.shuffle(indices)
    eval_indices = set(indices[:eval_count])

    train_split: list[dict[str, Any]] = []
    eval_split: list[dict[str, Any]] = []
    for idx, item in enumerate(train_data):
        if idx in eval_indices:
            eval_split.append(item)
        else:
            train_split.append(item)

    return train_split, eval_split


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT warm-up training for NL2SQL.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to configs/sft.yaml")
    return p


def _parse_overrides(extra: list[str]) -> dict[str, Any]:
    """Parse ``--key value`` pairs from the remaining CLI args."""
    overrides: dict[str, Any] = {}
    it = iter(extra)
    for token in it:
        if token.startswith("--"):
            key = token.lstrip("-").replace("-", "_")
            try:
                raw = next(it)
            except StopIteration:
                raise ValueError(f"Missing value for flag: {token}")
            # Best-effort type inference
            for cast in (int, float):
                try:
                    raw = cast(raw); break  # noqa: E702
                except ValueError:
                    pass
            if raw in ("true", "True"):
                raw = True
            elif raw in ("false", "False"):
                raw = False
            overrides[key] = raw
    return overrides


def main(argv: list[str] | None = None) -> int:
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}")

    args, extra = build_arg_parser().parse_known_args(argv)
    overrides = _parse_overrides(extra)
    cfg       = load_config(args.config, overrides)

    # ── tokenizer & model ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    )

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ── dataset ────────────────────────────────────────────────────────────
    train_data = json.loads(Path(cfg.train_data_path).read_text(encoding="utf-8"))
    train_records, eval_records = split_train_eval_data(train_data, cfg.eval_ratio, cfg.seed)
    print(
        f"[SFT] Raw split: train={len(train_records)} eval={len(eval_records)} "
        f"total={len(train_data)} eval_ratio={cfg.eval_ratio}"
    )

    train_dataset = build_sft_dataset(
        train_records,
        Path(cfg.database_root),
        tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    if len(train_dataset) == 0:
        raise ValueError("SFT train dataset is empty after preprocessing.")

    eval_dataset = None
    if eval_records:
        candidate_eval_dataset = build_sft_dataset(
            eval_records,
            Path(cfg.database_root),
            tokenizer,
            max_seq_length=cfg.max_seq_length,
        )
        if len(candidate_eval_dataset) > 0:
            eval_dataset = candidate_eval_dataset
        else:
            print("[SFT] Eval split became empty after preprocessing; disable eval for this run.")

    # ── training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    raise SystemExit(main())
