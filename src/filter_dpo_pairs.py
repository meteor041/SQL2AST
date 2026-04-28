#!/usr/bin/env python3
"""Filter noisy DPO pairs with SQL- and length-based heuristics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover
    Tokenizer = None  # type: ignore[assignment]


@dataclass
class PairRecord:
    raw: dict[str, Any]
    prompt_key: tuple[Any, Any] | str
    margin: float
    pair_type: str
    chosen_is_correct: bool | None
    rejected_is_correct: bool | None
    normalized_chosen: str
    normalized_rejected: str
    prompt_tokens: int = 0
    chosen_tokens: int = 0
    rejected_tokens: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter noisy DPO preference pairs.")
    parser.add_argument("--input", type=Path, required=True, help="Input DPO JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output JSONL.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON path for a machine-readable summary.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="sqlite",
        help="SQL dialect used for sqlglot parsing/serialization.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer directory or tokenizer.json used to enforce token-length limits.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Drop pairs with margin below this threshold.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=768,
        help="Drop pairs whose prompt token length exceeds this limit.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=1024,
        help="Drop pairs whose prompt+chosen or prompt+rejected token length exceeds this limit.",
    )
    parser.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=8,
        help="Keep at most this many pairs per prompt. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--allowed-pair-types",
        type=str,
        default="",
        help="Comma-separated pair types to keep, e.g. correct_wrong,wrong_wrong. Empty means keep all.",
    )
    parser.add_argument(
        "--min-margin-correct-wrong",
        type=float,
        default=None,
        help="Optional margin threshold override for correct_wrong pairs.",
    )
    parser.add_argument(
        "--min-margin-wrong-wrong",
        type=float,
        default=None,
        help="Optional margin threshold override for wrong_wrong pairs.",
    )
    parser.add_argument(
        "--require-chosen-correct",
        action="store_true",
        help="If set, keep only pairs whose chosen side is execution-correct.",
    )
    parser.add_argument(
        "--require-rejected-wrong",
        action="store_true",
        help="If set, keep only pairs whose rejected side is execution-wrong.",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_path: Path) -> Any:
    if Tokenizer is None:
        raise SystemExit("Missing dependency: tokenizers. Install it or omit --tokenizer.")

    resolved = tokenizer_path
    if tokenizer_path.is_dir():
        resolved = tokenizer_path / "tokenizer.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {resolved}")

    tokenizer = Tokenizer.from_file(str(resolved))
    try:
        tokenizer.no_truncation()
    except Exception:
        pass
    return tokenizer


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def normalize_sql(sql: str, dialect: str) -> str:
    expr = sqlglot.parse_one(sql, read=dialect)
    return expr.sql(dialect=dialect)


def prompt_key_for(record: dict[str, Any]) -> tuple[Any, Any] | str:
    sample_id = record.get("sample_id")
    db_id = record.get("db_id")
    if sample_id is not None or db_id is not None:
        return (sample_id, db_id)
    return str(record.get("prompt", ""))


def parse_allowed_pair_types(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def margin_threshold_for(pair_type: str, args: argparse.Namespace) -> float:
    if pair_type == "correct_wrong" and args.min_margin_correct_wrong is not None:
        return args.min_margin_correct_wrong
    if pair_type == "wrong_wrong" and args.min_margin_wrong_wrong is not None:
        return args.min_margin_wrong_wrong
    return args.min_margin


def select_pairs(
    records: list[PairRecord],
    max_pairs_per_prompt: int,
) -> tuple[list[PairRecord], int]:
    grouped: dict[tuple[Any, Any] | str, list[PairRecord]] = defaultdict(list)
    for record in records:
        grouped[record.prompt_key].append(record)

    selected: list[PairRecord] = []
    dropped = 0
    for pairs in grouped.values():
        pairs.sort(
            key=lambda item: (
                -item.margin,
                item.prompt_tokens + item.chosen_tokens + item.rejected_tokens,
            )
        )
        if max_pairs_per_prompt > 0:
            selected.extend(pairs[:max_pairs_per_prompt])
            dropped += max(len(pairs) - max_pairs_per_prompt, 0)
        else:
            selected.extend(pairs)
    return selected, dropped


def summarize_margins(records: list[PairRecord]) -> dict[str, float] | dict[str, int]:
    if not records:
        return {"count": 0}

    margins = sorted(record.margin for record in records)

    def q(frac: float) -> float:
        idx = int((len(margins) - 1) * frac)
        return margins[idx]

    return {
        "count": len(margins),
        "min": margins[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": margins[-1],
    }


def main() -> int:
    args = parse_args()
    allowed_pair_types = parse_allowed_pair_types(args.allowed_pair_types)

    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer)

    kept_records: list[PairRecord] = []
    seen_pairs: set[tuple[tuple[Any, Any] | str, str, str]] = set()
    reason_counts: Counter[str] = Counter()
    total = 0
    unique_prompts_before: set[tuple[Any, Any] | str] = set()

    with args.input.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                reason_counts["invalid_json"] += 1
                continue

            prompt = str(record.get("prompt", "")).rstrip() + "\n"
            chosen = str(record.get("chosen", "")).strip()
            rejected = str(record.get("rejected", "")).strip()
            prompt_key = prompt_key_for(record)
            unique_prompts_before.add(prompt_key)
            pair_type = str(record.get("pair_type", "") or "")
            chosen_is_correct = record.get("chosen_is_correct")
            rejected_is_correct = record.get("rejected_is_correct")

            try:
                margin = float(record.get("margin", 0.0))
            except (TypeError, ValueError):
                reason_counts["invalid_margin"] += 1
                continue

            if allowed_pair_types and pair_type not in allowed_pair_types:
                reason_counts["pair_type_filtered"] += 1
                continue

            if args.require_chosen_correct and chosen_is_correct is not True:
                reason_counts["chosen_not_correct"] += 1
                continue
            if args.require_rejected_wrong and rejected_is_correct is not False:
                reason_counts["rejected_not_wrong"] += 1
                continue

            min_margin = margin_threshold_for(pair_type, args)
            if margin <= 0.0 and min_margin > 0.0:
                reason_counts["zero_margin"] += 1
                continue
            if margin < min_margin:
                reason_counts["low_margin"] += 1
                continue

            try:
                normalized_chosen = normalize_sql(chosen, args.dialect)
                normalized_rejected = normalize_sql(rejected, args.dialect)
            except Exception:
                reason_counts["parse_fail"] += 1
                continue

            if normalized_chosen == normalized_rejected:
                reason_counts["same_normalized_sql"] += 1
                continue

            dedupe_key = (prompt_key, normalized_chosen, normalized_rejected)
            if dedupe_key in seen_pairs:
                reason_counts["duplicate_normalized_pair"] += 1
                continue
            seen_pairs.add(dedupe_key)

            prompt_tokens = 0
            chosen_tokens = 0
            rejected_tokens = 0
            if tokenizer is not None:
                prompt_tokens = token_len(tokenizer, prompt)
                chosen_tokens = token_len(tokenizer, chosen)
                rejected_tokens = token_len(tokenizer, rejected)

                if prompt_tokens > args.max_prompt_tokens:
                    reason_counts["prompt_too_long"] += 1
                    continue
                if prompt_tokens + chosen_tokens > args.max_total_tokens:
                    reason_counts["chosen_too_long"] += 1
                    continue
                if prompt_tokens + rejected_tokens > args.max_total_tokens:
                    reason_counts["rejected_too_long"] += 1
                    continue

            kept_records.append(
                PairRecord(
                    raw=record,
                    prompt_key=prompt_key,
                    margin=margin,
                    pair_type=pair_type,
                    chosen_is_correct=chosen_is_correct if isinstance(chosen_is_correct, bool) else None,
                    rejected_is_correct=rejected_is_correct if isinstance(rejected_is_correct, bool) else None,
                    normalized_chosen=normalized_chosen,
                    normalized_rejected=normalized_rejected,
                    prompt_tokens=prompt_tokens,
                    chosen_tokens=chosen_tokens,
                    rejected_tokens=rejected_tokens,
                )
            )

    selected_records, capped_drops = select_pairs(
        kept_records,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
    )
    reason_counts["per_prompt_cap"] += capped_drops

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in selected_records:
            fh.write(json.dumps(record.raw, ensure_ascii=False) + "\n")

    unique_prompts_after = {record.prompt_key for record in selected_records}
    kept_pair_type_counts = Counter(record.pair_type for record in selected_records)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "total_input_pairs": total,
        "total_output_pairs": len(selected_records),
        "retention_ratio": round(len(selected_records) / total, 4) if total else 0.0,
        "unique_prompts_before": len(unique_prompts_before),
        "unique_prompts_after": len(unique_prompts_after),
        "kept_pair_type_counts": dict(kept_pair_type_counts),
        "reasons": dict(reason_counts),
        "filters": {
            "min_margin": args.min_margin,
            "min_margin_correct_wrong": args.min_margin_correct_wrong,
            "min_margin_wrong_wrong": args.min_margin_wrong_wrong,
            "max_prompt_tokens": args.max_prompt_tokens if tokenizer is not None else None,
            "max_total_tokens": args.max_total_tokens if tokenizer is not None else None,
            "max_pairs_per_prompt": args.max_pairs_per_prompt,
            "tokenizer": str(args.tokenizer) if args.tokenizer is not None else None,
            "dialect": args.dialect,
            "allowed_pair_types": sorted(allowed_pair_types),
            "require_chosen_correct": args.require_chosen_correct,
            "require_rejected_wrong": args.require_rejected_wrong,
        },
        "kept_margin_summary": summarize_margins(selected_records),
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
