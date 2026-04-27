"""Build DPO preference pairs from eval_results.

Reads ``*_eval.json`` files produced by ``eval.py``, computes
``hierarchical_distance`` for every candidate SQL against the gold SQL,
then constructs (chosen, rejected, margin) pairs and writes them as JSONL.

Usage
-----
python src/build_pairs.py \\
    --eval-dir eval_results \\
    --train-data /data/bird/train/train.json \\
    --database-root /data/bird/train/train_databases \\
    --output data/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distance.composite import hierarchical_distance
from src.utils.schema import load_schema, schema_to_prompt_dict
from src.utils.prompt import format_nl2sql_prompt, format_sql_response

EVAL_FILE_RE = re.compile(r"^(\d+)_.*_eval\.json$")


@dataclass
class DPOPair:
    prompt:    str
    chosen:    str
    rejected:  str
    margin:    float
    sample_id: int
    db_id:     str


# ── file discovery ────────────────────────────────────────────────────────────

def discover_eval_files(eval_dir: Path) -> list[Path]:
    """Return *_eval.json paths sorted by sample_id."""
    files: list[tuple[int, Path]] = []
    for path in eval_dir.glob("*_eval.json"):
        m = EVAL_FILE_RE.match(path.name)
        if m:
            files.append((int(m.group(1)), path))
    files.sort()
    return [p for _, p in files]


def load_eval_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("correct_set", "wrong_set", "metadata"):
        if key not in data:
            raise ValueError(f"Missing '{key}' in {path.name}")
    return data


def load_train_item(train_data: list[dict], sample_id: int) -> dict | None:
    if 0 <= sample_id < len(train_data):
        return train_data[sample_id]
    return None


def _resolve_db_path(database_root: Path, db_id: str) -> Path:
    nested = database_root / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"SQLite DB not found: {nested}")


# ── scoring and pairing ───────────────────────────────────────────────────────

def score_sql_candidates(
    correct_sqls: list[str],
    wrong_sqls: list[str],
    gold_sql: str,
    schema: Any,
    dialect: str = "sqlite",
) -> list[tuple[str, float, bool]]:
    """Return [(sql, distance, is_correct), ...] sorted by distance ascending."""
    results: list[tuple[str, float, bool]] = []
    for sql in correct_sqls:
        d = hierarchical_distance(sql, gold_sql, schema=schema, dialect=dialect)
        results.append((sql, d, True))
    for sql in wrong_sqls:
        d = hierarchical_distance(sql, gold_sql, schema=schema, dialect=dialect)
        results.append((sql, d, False))
    results.sort(key=lambda x: x[1])
    return results


def build_pairs_for_sample(
    scored: list[tuple[str, float, bool]],
    gold_sql: str,
    prompt_str: str,
    sample_id: int,
    db_id: str,
    max_pairs: int = 0,
    min_margin: float = 0.0,
) -> list[DPOPair]:
    """Construct DPO pairs from scored candidates.

    Strategy
    --------
    - correct → wrong: pair every correct SQL with every wrong SQL
    - wrong → wrong: pair lower-distance wrong SQL with higher-distance wrong SQL
    - correct → correct: skip because execution cannot decide a preference
    - Deduplicate by (chosen_sql, rejected_sql)
    - Return every pair unless max_pairs > 0, then keep the largest margins
    """
    if not scored:
        return []

    pairs: list[DPOPair] = []
    seen:  set[tuple[str, str]] = set()

    def add_pair(chosen_sql: str, rejected_sql: str, margin: float) -> None:
        if chosen_sql == rejected_sql:
            return
        key = (chosen_sql, rejected_sql)
        if key in seen:
            return
        seen.add(key)
        pairs.append(DPOPair(
            prompt=prompt_str,
            chosen=format_sql_response(chosen_sql),
            rejected=format_sql_response(rejected_sql),
            margin=round(max(margin, 0.0), 4),
            sample_id=sample_id,
            db_id=db_id,
        ))

    correct = [(sql, dist) for sql, dist, is_correct in scored if is_correct]
    wrong = [(sql, dist) for sql, dist, is_correct in scored if not is_correct]

    for correct_sql, correct_dist in correct:
        for wrong_sql, wrong_dist in wrong:
            add_pair(correct_sql, wrong_sql, wrong_dist - correct_dist)

    wrong_sorted = sorted(wrong, key=lambda item: item[1])
    for i, (chosen_sql, chosen_dist) in enumerate(wrong_sorted):
        for rejected_sql, rejected_dist in wrong_sorted[i + 1:]:
            margin = rejected_dist - chosen_dist
            if margin <= 0.0 or margin < min_margin:
                continue
            add_pair(chosen_sql, rejected_sql, margin)

    pairs.sort(key=lambda p: p.margin, reverse=True)
    if max_pairs > 0:
        return pairs[:max_pairs]
    return pairs


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_jsonl(pairs: list[DPOPair], output_path: Path) -> int:
    """Append pairs to a JSONL file.  Returns number of lines written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        for p in pairs:
            record = {
                "prompt":    p.prompt,
                "chosen":    p.chosen,
                "rejected":  p.rejected,
                "margin":    p.margin,
                "sample_id": p.sample_id,
                "db_id":     p.db_id,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(pairs)


# ── per-file processing ───────────────────────────────────────────────────────

def process_eval_file(
    eval_path: Path,
    train_data: list[dict],
    database_root: Path,
    max_pairs: int,
    min_margin: float,
    dialect: str,
) -> tuple[list[DPOPair], str | None]:
    """Process one eval file.  Returns (pairs, error_message)."""
    try:
        data = load_eval_result(eval_path)
    except Exception as exc:
        return [], f"load error: {exc}"

    metadata  = data.get("metadata", {})
    sample_id = metadata.get("sample_id")
    db_id     = metadata.get("db_id")

    if not isinstance(sample_id, int):
        return [], "missing sample_id"

    train_item = load_train_item(train_data, sample_id)
    if train_item is None:
        return [], f"sample_id {sample_id} out of range"

    if not isinstance(db_id, str):
        for record in data.get("correct_set", []) + data.get("wrong_set", []):
            candidate_db_id = record.get("db_id")
            if isinstance(candidate_db_id, str):
                db_id = candidate_db_id
                break
    if not isinstance(db_id, str):
        candidate_db_id = train_item.get("db_id")
        if isinstance(candidate_db_id, str):
            db_id = candidate_db_id

    if not isinstance(sample_id, int) or not isinstance(db_id, str):
        return [], "missing sample_id or db_id"

    question = train_item.get("question", "")
    evidence = train_item.get("evidence", "")

    try:
        db_path = _resolve_db_path(database_root, db_id)
        schema  = load_schema(db_path)
    except Exception as exc:
        return [], f"schema load error: {exc}"

    # Gold SQL lives in every record's gold_sql field
    gold_sql: str | None = None
    for record in data.get("correct_set", []) + data.get("wrong_set", []):
        if isinstance(record.get("gold_sql"), str):
            gold_sql = record["gold_sql"]
            break
    if not gold_sql and isinstance(train_item.get("SQL"), str):
        gold_sql = train_item["SQL"]
    if not gold_sql:
        return [], "no gold_sql found"

    correct_sqls = [r["sql"] for r in data.get("correct_set", []) if isinstance(r.get("sql"), str)]
    wrong_sqls   = [r["sql"] for r in data.get("wrong_set",   []) if isinstance(r.get("sql"), str)]

    scored = score_sql_candidates(correct_sqls, wrong_sqls, gold_sql, schema, dialect)

    schema_dict = schema_to_prompt_dict(schema)
    prompt_str  = format_nl2sql_prompt(
        question, schema_dict, evidence, db_engine=dialect
    )

    pairs = build_pairs_for_sample(
        scored, gold_sql, prompt_str, sample_id, db_id, max_pairs, min_margin,
    )
    return pairs, None


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build DPO preference pairs from eval results.")
    p.add_argument("--eval-dir",       type=Path, default=Path("eval_results"))
    p.add_argument("--output",         type=Path, default=Path("data/dpo_pairs.jsonl"))
    p.add_argument("--train-data",     type=Path, required=True)
    p.add_argument("--database-root",  type=Path, required=True)
    p.add_argument("--max-pairs",      type=int,  default=0,
                   help="Maximum pairs per sample; 0 means keep all pairable pairs.")
    p.add_argument("--min-margin",     type=float, default=0.0,
                   help="Minimum distance gap for wrong-vs-wrong pairs.")
    p.add_argument("--dialect",        default="sqlite")
    p.add_argument("--limit",          type=int,  default=None,
                   help="Process only the first N eval files (for debugging).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    train_data = json.loads(Path(args.train_data).read_text(encoding="utf-8"))
    eval_files = discover_eval_files(args.eval_dir)
    if args.limit:
        eval_files = eval_files[: args.limit]

    # Start fresh
    if args.output.exists():
        args.output.unlink()

    total_pairs = 0
    errors: list[dict] = []

    for idx, eval_path in enumerate(eval_files, 1):
        print(f"[{idx}/{len(eval_files)}] {eval_path.name}", file=sys.stderr)
        pairs, err = process_eval_file(
            eval_path, train_data, args.database_root,
            args.max_pairs, args.min_margin, args.dialect,
        )
        if err:
            errors.append({"file": str(eval_path), "error": err})
        else:
            total_pairs += write_jsonl(pairs, args.output)

    summary = {
        "total_pairs":  total_pairs,
        "total_files":  len(eval_files),
        "error_count":  len(errors),
        "output":       str(args.output),
        "errors":       errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    raise SystemExit(main())
