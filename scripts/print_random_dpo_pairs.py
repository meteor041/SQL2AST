#!/usr/bin/env python3
"""Print random samples from the DPO pairs JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def _load_dpo_path(config_path: Path) -> Path:
    if yaml is None:
        raise SystemExit("Missing dependency: PyYAML. Run `pip install pyyaml` or pass --path directly.")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    dpo_pairs_path = config.get("dpo_pairs_path")
    if not dpo_pairs_path:
        raise SystemExit(f"Missing `dpo_pairs_path` in {config_path}")

    return Path(dpo_pairs_path).expanduser()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


def _print_pair(index: int, total: int, record: dict[str, Any]) -> None:
    print("=" * 100)
    print(f"[{index}/{total}] sample_id={record.get('sample_id', '-')} db_id={record.get('db_id', '-')}")
    print(f"margin={record.get('margin', '-')}")
    print()
    print("[PROMPT]")
    print(record.get("prompt", ""))
    print()
    print("[CHOSEN]")
    print(record.get("chosen", ""))
    print()
    print("[REJECTED]")
    print(record.get("rejected", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomly print DPO pairs from JSONL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dpo.local.yaml"),
        help="YAML config containing dpo_pairs_path. Default: configs/dpo.local.yaml",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="DPO pairs JSONL path. Overrides --config.",
    )
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of pairs to print. Default: 5")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num <= 0:
        raise SystemExit("--num must be positive")

    path = args.path.expanduser() if args.path is not None else _load_dpo_path(args.config)
    if not path.exists():
        raise SystemExit(f"DPO pairs file does not exist: {path}")

    records = _read_jsonl(path)
    if not records:
        raise SystemExit(f"No records found in {path}")

    rng = random.Random(args.seed)
    samples = rng.sample(records, k=min(args.num, len(records)))

    print(f"Loaded {len(records)} DPO pairs from {path}")
    print(f"Printing {len(samples)} random pair(s)")
    for idx, record in enumerate(samples, start=1):
        _print_pair(idx, len(samples), record)
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
