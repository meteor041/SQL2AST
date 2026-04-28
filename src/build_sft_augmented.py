"""Build an augmented SFT dataset from execution-correct sampled SQL.

This script reads eval results from ``eval_results/``, keeps sampled SQL that:
- are execution-correct (from ``correct_set``)
- are normalized-SQL different from the gold SQL
- optionally rank near the gold SQL under the AST distance

It then emits either:
- augmented-only records, or
- a merged train.json-style list (original + augmented)

Each augmented record preserves the original SFT schema:
``db_id``, ``question``, ``evidence``, ``SQL``

Extra metadata fields are attached for provenance but ignored by ``train_sft.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distance.composite import hierarchical_distance
from src.normalize import normalize_sql
from src.utils.schema import load_schema, schema_to_prompt_dict


@dataclass
class AugmentedSample:
    record: dict[str, Any]
    distance: float
    count: int
    normalized_sql: str


def _resolve_db_path(database_root: Path, db_id: str) -> Path:
    nested = database_root / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    flat = database_root / f"{db_id}.sqlite"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"SQLite DB not found: {nested} or {flat}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment SFT train.json with execution-correct sampled SQL."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Merged output (original + augmented) in train.json format.")
    parser.add_argument("--augmented-only-output", type=Path, default=None,
                        help="Optional output for augmented-only records.")
    parser.add_argument("--dialect", type=str, default="sqlite")
    parser.add_argument("--max-augment-per-sample", type=int, default=2,
                        help="Keep at most this many augmented SQLs per original sample.")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Optional upper bound on AST distance to gold SQL.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum occurrence count for a sampled SQL cluster.")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def discover_eval_files(eval_dir: Path) -> list[Path]:
    files = sorted(eval_dir.glob("*_eval.json"))
    return files


def load_eval_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("correct_set", "wrong_set", "metadata"):
        if key not in data:
            raise ValueError(f"Missing '{key}' in {path.name}")
    return data


def build_augmented_samples(
    eval_files: list[Path],
    train_data: list[dict[str, Any]],
    database_root: Path,
    dialect: str,
    max_augment_per_sample: int,
    max_distance: float | None,
    min_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augmented_records: list[dict[str, Any]] = []
    schema_cache: dict[str, tuple[Any, dict[str, list[str]]]] = {}

    stats = {
        "total_eval_files": 0,
        "augmented_samples": 0,
        "samples_with_augmentation": 0,
        "skipped_same_as_gold": 0,
        "skipped_parse_or_distance_error": 0,
        "skipped_below_min_count": 0,
        "skipped_over_max_distance": 0,
        "deduped_correct_candidates": 0,
    }

    for eval_path in eval_files:
        stats["total_eval_files"] += 1
        data = load_eval_result(eval_path)
        metadata = data.get("metadata", {})
        sample_id = metadata.get("sample_id")
        if not isinstance(sample_id, int) or not (0 <= sample_id < len(train_data)):
            continue

        train_item = train_data[sample_id]
        db_id = train_item.get("db_id")
        question = train_item.get("question", "")
        evidence = train_item.get("evidence", "")
        gold_sql = train_item.get("SQL", "")
        if not isinstance(db_id, str) or not gold_sql:
            continue

        if db_id not in schema_cache:
            db_path = _resolve_db_path(database_root, db_id)
            raw_schema = load_schema(db_path)
            schema_cache[db_id] = (raw_schema, schema_to_prompt_dict(raw_schema))
        schema, schema_dict = schema_cache[db_id]

        try:
            normalized_gold = normalize_sql(gold_sql, schema=schema_dict, dialect=dialect)
        except Exception:
            normalized_gold = gold_sql.strip()

        unique_candidates: dict[str, AugmentedSample] = {}
        for record in data.get("correct_set", []):
            sql = record.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                continue

            count = int(record.get("count", 1) or 1)
            if count < min_count:
                stats["skipped_below_min_count"] += 1
                continue

            try:
                normalized_sql = normalize_sql(sql, schema=schema_dict, dialect=dialect)
            except Exception:
                stats["skipped_parse_or_distance_error"] += 1
                continue

            if normalized_sql == normalized_gold:
                stats["skipped_same_as_gold"] += 1
                continue

            try:
                distance = hierarchical_distance(sql, gold_sql, schema=schema, dialect=dialect)
            except Exception:
                stats["skipped_parse_or_distance_error"] += 1
                continue

            if max_distance is not None and distance > max_distance:
                stats["skipped_over_max_distance"] += 1
                continue

            existing = unique_candidates.get(normalized_sql)
            if existing is None:
                unique_candidates[normalized_sql] = AugmentedSample(
                    record=record,
                    distance=distance,
                    count=count,
                    normalized_sql=normalized_sql,
                )
            else:
                stats["deduped_correct_candidates"] += 1
                better = (
                    distance < existing.distance or
                    (distance == existing.distance and count > existing.count)
                )
                if better:
                    unique_candidates[normalized_sql] = AugmentedSample(
                        record=record,
                        distance=distance,
                        count=count,
                        normalized_sql=normalized_sql,
                    )

        ranked = sorted(
            unique_candidates.values(),
            key=lambda item: (item.distance, -item.count, len(item.record["sql"])),
        )
        ranked = ranked[:max_augment_per_sample] if max_augment_per_sample > 0 else ranked

        if ranked:
            stats["samples_with_augmentation"] += 1

        for rank, item in enumerate(ranked, start=1):
            augmented_records.append({
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "SQL": item.record["sql"],
                "source": "augmented_correct_sample",
                "sample_id": sample_id,
                "augmentation_rank": rank,
                "augmentation_distance": round(item.distance, 4),
                "augmentation_count": item.count,
                "gold_sql": gold_sql,
            })
            stats["augmented_samples"] += 1

    return augmented_records, stats


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    train_data = json.loads(args.train_data.read_text(encoding="utf-8"))
    eval_files = discover_eval_files(args.eval_dir)
    if args.limit is not None:
        eval_files = eval_files[:args.limit]

    augmented_records, stats = build_augmented_samples(
        eval_files=eval_files,
        train_data=train_data,
        database_root=args.database_root,
        dialect=args.dialect,
        max_augment_per_sample=args.max_augment_per_sample,
        max_distance=args.max_distance,
        min_count=args.min_count,
    )

    merged_records = list(train_data) + augmented_records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.augmented_only_output is not None:
        args.augmented_only_output.parent.mkdir(parents=True, exist_ok=True)
        args.augmented_only_output.write_text(
            json.dumps(augmented_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "train_input_count": len(train_data),
        "augmented_count": len(augmented_records),
        "merged_count": len(merged_records),
        "output": str(args.output),
        "augmented_only_output": str(args.augmented_only_output) if args.augmented_only_output else None,
        "filters": {
            "max_augment_per_sample": args.max_augment_per_sample,
            "max_distance": args.max_distance,
            "min_count": args.min_count,
            "dialect": args.dialect,
        },
        "stats": stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
