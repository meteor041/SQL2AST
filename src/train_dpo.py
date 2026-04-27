"""Margin-aware DPO training.

Loss formula:
    L = -log σ( β*(r_w − r_l) + α*margin )

where margin = D(rejected, gold) − D(chosen, gold) and α scales the
contribution of the AST-distance margin to the loss.

Usage
-----
python src/train_dpo.py --config configs/dpo.yaml
python src/train_dpo.py --config configs/dpo.yaml --beta 0.2 --alpha 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: PyYAML.  Run `pip install pyyaml`.")


@dataclass
class DPOTrainConfig:
    # ── model ──────────────────────────────────────────────────────────────
    model_name_or_path:     str        = "outputs/sft"
    ref_model_name_or_path: str | None = None   # None → frozen copy of policy
    # ── data ───────────────────────────────────────────────────────────────
    dpo_pairs_path: str = "data/dpo_pairs.jsonl"
    output_dir:     str = "outputs/dpo"
    # ── DPO hyper-parameters ───────────────────────────────────────────────
    beta:  float = 0.1   # KL penalty coefficient
    alpha: float = 1.0   # margin scaling coefficient
    # ── training ───────────────────────────────────────────────────────────
    num_train_epochs:            int   = 1
    per_device_train_batch_size: int   = 2
    gradient_accumulation_steps: int   = 8
    learning_rate:               float = 5e-6
    warmup_ratio:                float = 0.1
    lr_scheduler_type:           str   = "cosine"
    max_seq_length:              int   = 2048
    max_prompt_length:           int   = 1536
    eval_split:                  float = 0.05   # fraction used for evaluation
    # ── LoRA ───────────────────────────────────────────────────────────────
    use_lora:            bool       = True
    lora_r:              int        = 16
    lora_alpha:          int        = 32
    lora_dropout:        float      = 0.05
    lora_target_modules: list[str] | None = None
    # ── misc ───────────────────────────────────────────────────────────────
    seed:                   int  = 42
    bf16:                   bool = True
    fp16:                   bool = False
    logging_steps:          int  = 10
    save_steps:             int  = 100
    dataloader_num_workers: int  = 2


def load_config(config_path: Path, overrides: dict[str, Any] | None = None) -> DPOTrainConfig:
    with config_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    if overrides:
        data.update(overrides)
    valid = {f.name for f in fields(DPOTrainConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return DPOTrainConfig(**{k: v for k, v in data.items() if k in valid})


def load_dpo_dataset(
    pairs_path: Path,
    tokenizer: Any,
    max_prompt_length: int = 1536,
    max_seq_length: int = 2048,
    eval_split: float = 0.05,
) -> tuple[Any, Any]:
    """Load preference pairs JSONL into (train_dataset, eval_dataset).

    Each line must have: prompt, chosen, rejected, margin.
    The margin is stored as a float feature and passed to the trainer.
    """
    from datasets import Dataset

    records: list[dict] = []
    with pairs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                record["prompt"] = str(record["prompt"]).rstrip() + "\n"
                record["chosen"] = str(record["chosen"]).strip()
                record["rejected"] = str(record["rejected"]).strip()
                records.append(record)

    dataset = Dataset.from_list(records)

    # TRL DPOTrainer expects the raw text fields; tokenization happens internally.
    # We keep 'margin' as an extra float column.
    split   = dataset.train_test_split(test_size=eval_split, seed=42)
    return split["train"], split["test"]


# ── Margin-aware DPO trainer ──────────────────────────────────────────────────

class MarginAwareDPOTrainer:
    """Factory function that returns a patched DPOTrainer subclass at runtime.

    Defined as a factory to avoid importing torch/trl at module load time.
    Call ``MarginAwareDPOTrainer.make(alpha)`` to obtain the class.
    """

    @staticmethod
    def make(alpha: float = 1.0) -> type:
        import torch
        import torch.nn.functional as F
        from trl import DPOTrainer
        from trl.trainer.dpo_trainer import (
            disable_gradient_checkpointing,
            entropy_from_logits,
            is_peft_model,
            selective_log_softmax,
            use_adapter,
        )

        class _MarginAwareDPOTrainer(DPOTrainer):
            """DPOTrainer with margin-scaled loss.

            Loss:  L = -log σ( β*(r_w − r_l) + α*margin )

            The ``margin`` tensor is extracted from the batch inside
            ``compute_loss`` and stashed in ``self._current_margin`` for
            ``dpo_loss`` to consume.
            """

            _alpha: float = alpha

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: int | None = None,
            ) -> Any:
                return super().compute_loss(
                    model, inputs,
                    return_outputs=return_outputs,
                    **({"num_items_in_batch": num_items_in_batch}
                       if num_items_in_batch is not None else {}),
                )

            def _compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool) -> Any:
                """TRL 1.2.0 sigmoid DPO loss with an added AST-distance margin."""
                if any(loss_type != "sigmoid" for loss_type in self.loss_types):
                    raise ValueError("MarginAwareDPOTrainer currently supports only sigmoid DPO loss.")

                mode = "train" if self.model.training else "eval"
                margin = inputs.pop("margin", None)
                if margin is not None and not isinstance(margin, torch.Tensor):
                    margin = torch.tensor(margin, dtype=torch.float32)

                non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
                model_kwargs = {k: v for k, v in inputs.items() if k not in non_model_keys}
                model_kwargs["use_cache"] = False
                outputs = model(**model_kwargs)

                input_ids = inputs["input_ids"]
                completion_mask = inputs["completion_mask"]
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_completion_mask = completion_mask[..., 1:].contiguous()

                per_token_logps = selective_log_softmax(shift_logits, shift_labels)
                per_token_logps[shift_completion_mask == 0] = 0.0
                chosen_logps, rejected_logps = per_token_logps.sum(dim=1).chunk(2, dim=0)

                if self.precompute_ref_logps:
                    ref_chosen_logps = inputs["ref_chosen_logps"]
                    ref_rejected_logps = inputs["ref_rejected_logps"]
                else:
                    with torch.no_grad(), disable_gradient_checkpointing(
                        self.model, self.args.gradient_checkpointing_kwargs
                    ):
                        if is_peft_model(model) and self.ref_model is None:
                            unwrapped_model = self.accelerator.unwrap_model(model)
                            with use_adapter(
                                unwrapped_model,
                                adapter_name="ref" if "ref" in unwrapped_model.peft_config else None,
                            ):
                                ref_outputs = self.model(**model_kwargs)
                        else:
                            ref_outputs = self.ref_model(**model_kwargs)

                    ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
                    ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
                    ref_per_token_logps[shift_completion_mask == 0] = 0.0
                    ref_chosen_logps, ref_rejected_logps = ref_per_token_logps.sum(dim=1).chunk(2, dim=0)

                chosen_logratios = chosen_logps - ref_chosen_logps
                rejected_logratios = rejected_logps - ref_rejected_logps
                delta_score = chosen_logratios - rejected_logratios

                logits = self.beta * delta_score
                if margin is not None:
                    logits = logits + self._alpha * margin.to(logits.device, dtype=logits.dtype)

                loss = -F.logsigmoid(logits).mean()

                entropy = entropy_from_logits(shift_logits.detach())
                entropy = entropy[shift_completion_mask.bool()].mean()
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
                self._metrics[mode]["entropy"].append(entropy)

                if mode == "train":
                    num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
                    self._total_train_tokens += num_tokens
                self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

                chosen_rewards = self.beta * chosen_logratios.detach()
                rejected_rewards = self.beta * rejected_logratios.detach()
                agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
                agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
                self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
                self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())
                reward_accuracies = (chosen_rewards > rejected_rewards).float()
                self._metrics[mode]["rewards/accuracies"].append(
                    self.accelerator.gather(reward_accuracies).mean().item()
                )
                reward_margins = chosen_rewards - rejected_rewards
                self._metrics[mode]["rewards/margins"].append(
                    self.accelerator.gather(reward_margins).mean().item()
                )
                self._metrics[mode]["logps/chosen"].append(
                    self.accelerator.gather(chosen_logps).mean().item()
                )
                self._metrics[mode]["logps/rejected"].append(
                    self.accelerator.gather(rejected_logps).mean().item()
                )
                if margin is not None:
                    self._metrics[mode]["margin/ast"].append(
                        self.accelerator.gather(margin.to(chosen_rewards.device)).mean().item()
                    )

                return (loss, outputs) if return_outputs else loss

        return _MarginAwareDPOTrainer


# ── CLI helpers ───────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Margin-aware DPO training.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to configs/dpo.yaml")
    return p


def _parse_overrides(extra: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    it = iter(extra)
    for token in it:
        if token.startswith("--"):
            key = token.lstrip("-").replace("-", "_")
            try:
                raw = next(it)
            except StopIteration:
                raise ValueError(f"Missing value for flag: {token}")
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


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import DPOConfig
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc}")

    args, extra = build_arg_parser().parse_known_args(argv)
    overrides = _parse_overrides(extra)
    cfg       = load_config(args.config, overrides)

    # ── tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── policy model ───────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
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

    # ── reference model (None = frozen copy of policy) ─────────────────────
    ref_model = None
    if cfg.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_name_or_path,
            dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            trust_remote_code=True,
        )

    # ── dataset ────────────────────────────────────────────────────────────
    train_dataset, eval_dataset = load_dpo_dataset(
        Path(cfg.dpo_pairs_path),
        tokenizer,
        max_prompt_length=cfg.max_prompt_length,
        max_seq_length=cfg.max_seq_length,
        eval_split=cfg.eval_split,
    )

    # ── DPO config ─────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        beta=cfg.beta,
        max_length=cfg.max_seq_length,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    TrainerClass = MarginAwareDPOTrainer.make(alpha=cfg.alpha)

    trainer = TrainerClass(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    raise SystemExit(main())
