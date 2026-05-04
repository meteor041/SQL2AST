"""Build csc_sql-compatible DB-content Lucene indexes for flat/nested CHES DBs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.cscsql_prompt import (
    _load_process_dataset_module,
    resolve_db_path,
    resolve_tables_json_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build csc_sql-compatible DB-content Lucene indexes."
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        required=True,
        help="Root containing flat or nested SQLite DBs.",
    )
    parser.add_argument(
        "--output-index-root",
        type=Path,
        default=None,
        help="Destination root for per-db Lucene indexes. "
        "If omitted, defaults to a sibling *_db_contents_index directory.",
    )
    parser.add_argument(
        "--tables-json",
        type=Path,
        default=None,
        help="Optional schema metadata JSON. Defaults to auto-detection.",
    )
    return parser


def infer_output_index_root(database_root: Path) -> Path:
    if database_root.name.endswith("_databases"):
        prefix = database_root.name[:-10]
        return database_root.parent / f"{prefix}_db_contents_index"
    return database_root.parent / f"{database_root.name}_db_contents_index"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    module = _load_process_dataset_module()

    tables_json = args.tables_json or resolve_tables_json_path(args.database_root)
    index_root = args.output_index_root or infer_output_index_root(args.database_root)
    index_root.mkdir(parents=True, exist_ok=True)

    db_info_map = {}
    for db_info in module.load_json_file(str(tables_json)):
        db_id = db_info.get("db_id")
        if isinstance(db_id, str):
            db_info_map[db_id] = db_info

    total = len(db_info_map)
    for idx, db_id in enumerate(sorted(db_info_map), 1):
        db_path = resolve_db_path(args.database_root, db_id)
        out_dir = index_root / db_id
        print(f"[{idx}/{total}] building index for {db_id}: {db_path} -> {out_dir}")
        module.remove_contents_of_a_folder(str(out_dir))
        module.build_content_index(str(db_path), str(out_dir))

    print(f"Finished building indexes under: {index_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
