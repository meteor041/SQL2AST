"""Calibrate the distance function via Spearman rank correlation.

Computes D(candidate, gold) for every candidate in eval_results and
correlates it with execution correctness (is_correct).  We expect:

    Spearman(D, 1 − EX)  ≥  0.65

i.e. higher distance predicts lower execution accuracy.
(Equivalently, Spearman(D, EX) ≤ −0.65.)

Usage
-----
python src/calibrate.py \\
    --eval-dir eval_results \\
    --train-data /data/bird/train/train.json \\
    --database-root /data/bird/train/train_databases \\
    --output reports/distance_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distance.composite import hierarchical_distance
from src.utils.schema import load_schema

EVAL_FILE_RE = re.compile(r"^(\d+)_.*_eval\.json$")


def _resolve_db_path(database_root: Path, db_id: str) -> Path:
    nested = database_root / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"DB not found: {nested}")


def load_distance_correctness_pairs(
    eval_dir: Path,
    train_data: list[dict],
    database_root: Path,
    dialect: str = "sqlite",
    limit: int | None = None,
) -> list[tuple[float, int]]:
    """Return [(distance, is_correct), ...] for all candidates in eval_dir.

    ``is_correct`` is 1 if the candidate is in ``correct_set``, else 0.
    """
    def _eval_file_id(path: Path) -> int:
        match = EVAL_FILE_RE.match(path.name)
        assert match is not None
        return int(match.group(1))

    eval_files = sorted(
        (p for p in eval_dir.glob("*_eval.json") if EVAL_FILE_RE.match(p.name)),
        key=_eval_file_id,
    )
    if limit:
        eval_files = eval_files[:limit]

    pairs: list[tuple[float, int]] = []

    for idx, path in enumerate(eval_files, 1):
        print(f"[{idx}/{len(eval_files)}] {path.name}", file=sys.stderr)
        try:
            data     = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            db_id    = metadata.get("db_id")
            if not isinstance(db_id, str):
                continue

            db_path = _resolve_db_path(database_root, db_id)
            schema  = load_schema(db_path)

            gold_sql: str | None = None
            for record in data.get("correct_set", []) + data.get("wrong_set", []):
                if isinstance(record.get("gold_sql"), str):
                    gold_sql = record["gold_sql"]
                    break
            if not gold_sql:
                continue

            for record in data.get("correct_set", []):
                sql = record.get("sql")
                if isinstance(sql, str):
                    d = hierarchical_distance(sql, gold_sql, schema=schema, dialect=dialect)
                    pairs.append((d, 1))

            for record in data.get("wrong_set", []):
                sql = record.get("sql")
                if isinstance(sql, str):
                    d = hierarchical_distance(sql, gold_sql, schema=schema, dialect=dialect)
                    pairs.append((d, 0))

        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr)

    return pairs


def spearman_correlation(distances: list[float], correctness: list[int]) -> tuple[float, float | None]:
    """Return (rho, p_value).  Falls back to a manual implementation if scipy absent."""
    try:
        from scipy.stats import spearmanr

        result: Any = spearmanr(distances, correctness)
        rho = _as_float(getattr(result, "statistic", result[0]))
        p_value = _as_float(getattr(result, "pvalue", result[1]))
        if math.isnan(rho):
            return 0.0, None if math.isnan(p_value) else p_value
        return rho, None if math.isnan(p_value) else p_value
    except ImportError:
        pass

    # Manual Spearman via rank differences
    n = len(distances)
    if n < 2:
        return 0.0, None

    def _rank(xs: list[float]) -> list[float]:
        sorted_xs = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_xs, 1):
            ranks[idx] = float(rank)
        return ranks

    r_d = _rank(distances)
    r_c = _rank([float(c) for c in correctness])
    d_sq = sum((rd - rc) ** 2 for rd, rc in zip(r_d, r_c))
    rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
    return rho, None


def _as_float(value: Any) -> float:
    """Convert scipy/numpy scalar-like values to a plain float for type checkers."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def compute_calibration_report(pairs: list[tuple[float, int]]) -> dict:
    """Return a calibration statistics dictionary."""
    if not pairs:
        return {"error": "no data"}

    distances   = [d for d, _ in pairs]
    correctness = [c for _, c in pairs]

    correct_dists = [d for d, c in pairs if c == 1]
    wrong_dists   = [d for d, c in pairs if c == 0]

    rho, p_value = spearman_correlation(distances, correctness)

    auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score
        # Predict "wrong" (label=1) using distance (higher = more likely wrong)
        auc = float(roc_auc_score([1 - c for c in correctness], distances))
    except Exception:
        pass

    # Target: Spearman(D, EX) ≤ -0.65  (distance anti-correlates with correctness)
    passed = rho <= -0.65

    return {
        "spearman_rho":      round(rho, 4),
        "p_value":           round(p_value, 6) if p_value is not None else None,
        "n_samples":         len(pairs),
        "n_correct":         len(correct_dists),
        "n_wrong":           len(wrong_dists),
        "mean_dist_correct": round(sum(correct_dists) / len(correct_dists), 4) if correct_dists else None,
        "mean_dist_wrong":   round(sum(wrong_dists)   / len(wrong_dists),   4) if wrong_dists   else None,
        "auc_roc":           round(auc, 4) if auc is not None else None,
        "target":            "spearman_rho <= -0.65",
        "pass":              passed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate distance function with Spearman ρ.")
    p.add_argument("--eval-dir",      type=Path, default=Path("eval_results"))
    p.add_argument("--train-data",    type=Path, required=True)
    p.add_argument("--database-root", type=Path, required=True)
    p.add_argument("--output",        type=Path, default=None,
                   help="Write JSON report to this path (stdout if omitted).")
    p.add_argument("--dialect",       default="sqlite")
    p.add_argument("--limit",         type=int, default=None,
                   help="Limit number of eval files processed (for debugging).")
    return p


def main(argv: list[str] | None = None) -> int:
    args       = build_arg_parser().parse_args(argv)
    train_data = json.loads(Path(args.train_data).read_text(encoding="utf-8"))

    pairs  = load_distance_correctness_pairs(
        args.eval_dir, train_data, args.database_root, args.dialect, args.limit,
    )
    report = compute_calibration_report(pairs)
    output = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    print(output)

    if not report.get("pass", False):
        print(
            f"\nWARNING: Spearman ρ = {report.get('spearman_rho')} "
            f"— target is ≤ -0.65.  Tune distance weights before proceeding.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
