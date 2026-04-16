"""Normalize SQL statements and store the canonical string form.

The primary input format is an eval result JSON file, such as
``eval_results/544_movies_4_eval.json``. For these files, the script reads
``gold_sql``, ``correct_set[*].sql``, and ``wrong_set[*].sql`` and writes one
JSON payload per query with ``gold``, ``correct``, and ``wrong`` lists
containing the original SQL and its normalized form.

The normalized SQL is produced by parsing with sqlglot and re-serializing,
which gives a consistent representation suitable for deduplication and for
re-parsing into a live Expression object when distance metrics are needed.

The legacy sampled SQL format is still supported: JSON records with an
``all_sqls`` list are preserved and receive a ``normalized_sqls`` list.

The CLI supports either one JSON file or a directory of JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import sqlglot
    from sqlglot import expressions as exp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "Missing dependency: sqlglot. Install it with `pip install -r requirements.txt`."
    ) from exc


def parse_sql(sql: str, dialect: str | None = None) -> dict[str, Any]:
    """Parse one SQL statement and return the original SQL plus its normalized form."""
    expression = sqlglot.parse_one(sql, read=dialect) if dialect else sqlglot.parse_one(sql)
    normalized_sql = expression.sql(dialect=dialect) if dialect else expression.sql()

    return {
        "sql": sql,
        "normalized_sql": normalized_sql,
    }


def parse_sql_safely(sql: str, dialect: str | None = None) -> dict[str, Any]:
    """Parse one SQL string without aborting the whole dataset on one failure."""
    try:
        return parse_sql(sql, dialect=dialect)
    except Exception as exc:  # sqlglot raises several parse/unsupported errors
        return {
            "sql": sql,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }


def first_string_value(items: list[dict[str, Any]], key: str) -> str | None:
    """Return the first string value for key from a list of dictionaries."""
    for item in items:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return None


def select_eval_records(
    data: dict[str, Any],
    set_name: str,
    deduplicate: bool = False,
) -> list[dict[str, Any]]:
    """Return validated eval records from a named set."""
    records = data.get(set_name, [])
    if not isinstance(records, list):
        raise ValueError(f"Expected `{set_name}` to be a list of records.")

    selected_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected every item in `{set_name}` to be an object.")

        sql = record.get("sql")
        if not isinstance(sql, str):
            raise ValueError(
                f"Expected `{set_name}[{index}].sql` to be a SQL string."
            )

        if deduplicate and sql in seen:
            continue
        seen.add(sql)
        selected_records.append(record)

    return selected_records


def parse_eval_records(
    records: list[dict[str, Any]],
    set_name: str,
    dialect: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse selected eval records into AST payloads and tagged errors."""
    parsed_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record in records:
        sql = record["sql"]
        parsed = parse_sql_safely(sql, dialect=dialect)
        if "error" in parsed:
            errors.append(
                {
                    "set": set_name,
                    **parsed,
                }
            )
        else:
            parsed_records.append(parsed)

    return parsed_records, errors


def select_gold_sqls(
    records: list[dict[str, Any]],
) -> list[str]:
    """Return unique gold SQL strings in first-seen order."""
    gold_sqls: list[str] = []
    seen: set[str] = set()
    for record in records:
        gold_sql = record.get("gold_sql")
        if not isinstance(gold_sql, str) or gold_sql in seen:
            continue
        seen.add(gold_sql)
        gold_sqls.append(gold_sql)
    return gold_sqls


def parse_gold_sqls(
    gold_sqls: list[str],
    dialect: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse gold SQL strings into AST payloads and tagged errors."""
    gold: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sql in gold_sqls:
        parsed = parse_sql_safely(sql, dialect=dialect)
        if "error" in parsed:
            errors.append(
                {
                    "set": "gold",
                    **parsed,
                }
            )
        else:
            gold.append(parsed)
    return gold, errors


def parse_eval_result(
    data: dict[str, Any],
    source_path: Path | None = None,
    dialect: str | None = None,
    deduplicate: bool = False,
) -> dict[str, Any]:
    """Parse ``correct_set[*].sql`` and ``wrong_set[*].sql`` from an eval result."""
    if "correct_set" not in data:
        raise ValueError("Expected eval result JSON to contain `correct_set`.")

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    selected_correct_records = select_eval_records(
        data,
        "correct_set",
        deduplicate=deduplicate,
    )
    selected_wrong_records = select_eval_records(
        data,
        "wrong_set",
        deduplicate=deduplicate,
    )

    selected_gold_sqls = select_gold_sqls(
        selected_correct_records + selected_wrong_records
    )

    gold, gold_errors = parse_gold_sqls(
        selected_gold_sqls,
        dialect=dialect,
    )
    correct, correct_errors = parse_eval_records(
        selected_correct_records,
        "correct_set",
        dialect=dialect,
    )
    wrong, wrong_errors = parse_eval_records(
        selected_wrong_records,
        "wrong_set",
        dialect=dialect,
    )
    errors = gold_errors + correct_errors + wrong_errors

    sample_id = metadata.get("sample_id")
    if sample_id is None:
        for records in (selected_correct_records, selected_wrong_records):
            if records:
                sample_id = records[0].get("sample_id")
                break

    db_id = metadata.get("db_id")
    if not isinstance(db_id, str):
        db_id = first_string_value(selected_correct_records, "db_id")
    if not isinstance(db_id, str):
        db_id = first_string_value(selected_wrong_records, "db_id")

    output_metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "db_id": db_id,
        "source_file": str(source_path) if source_path is not None else None,
        "gold_count": len(selected_gold_sqls),
        "correct_count": len(selected_correct_records),
        "wrong_count": len(selected_wrong_records),
        "parsed_gold_count": len(gold),
        "parsed_correct_count": len(correct),
        "parsed_wrong_count": len(wrong),
        "parsed_count": len(gold) + len(correct) + len(wrong),
        "error_count": len(errors),
    }

    return {
        "metadata": output_metadata,
        "gold": gold,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
    }


def parse_dataset(
    data: Any,
    dialect: str | None = None,
    deduplicate: bool = False,
) -> Any:
    """Add AST data to every record that has an ``all_sqls`` field."""
    if isinstance(data, list):
        return [
            parse_dataset(item, dialect=dialect, deduplicate=deduplicate)
            for item in data
        ]

    if not isinstance(data, dict) or "all_sqls" not in data:
        return data

    sqls = data["all_sqls"]
    if not isinstance(sqls, list):
        raise ValueError("Expected `all_sqls` to be a list of SQL strings.")

    selected_sqls: list[str] = []
    seen: set[str] = set()
    for sql in sqls:
        if not isinstance(sql, str):
            raise ValueError("Expected every item in `all_sqls` to be a string.")
        if deduplicate and sql in seen:
            continue
        seen.add(sql)
        selected_sqls.append(sql)

    parsed = {
        **data,
        "normalized_sqls": [
            parse_sql_safely(sql, dialect=dialect)
            for sql in selected_sqls
        ],
    }
    return parsed


def parse_json_data(
    data: Any,
    source_path: Path | None = None,
    dialect: str | None = None,
    deduplicate: bool = False,
) -> Any:
    """Dispatch between eval result input and legacy sampled SQL input."""
    if isinstance(data, dict) and "correct_set" in data:
        return parse_eval_result(
            data,
            source_path=source_path,
            dialect=dialect,
            deduplicate=deduplicate,
        )

    return parse_dataset(data, dialect=dialect, deduplicate=deduplicate)


def parse_json_file(
    input_path: Path,
    output_path: Path,
    dialect: str | None = None,
    deduplicate: bool = False,
) -> None:
    """Parse one JSON file and write the transformed dataset to output_path."""
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    parsed = parse_json_data(
        data,
        source_path=input_path,
        dialect=dialect,
        deduplicate=deduplicate,
    )
    output = json.dumps(parsed, ensure_ascii=False, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")


def iter_input_files(input_dir: Path, pattern: str) -> list[Path]:
    """Return source JSON files, excluding previous AST output by default."""
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and not path.name.endswith("_ast.json")
    )


def build_batch_output_path(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map input JSON files to AST outputs, preserving subdirectories."""
    relative_path = input_file.relative_to(input_dir)
    output_stem = relative_path.stem
    if output_stem.endswith("_eval"):
        output_stem = output_stem[: -len("_eval")]
    return output_dir / relative_path.with_name(f"{output_stem}_ast.json")


def parse_directory(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    dialect: str | None = None,
    deduplicate: bool = False,
) -> tuple[int, list[dict[str, str]]]:
    """Parse every matched JSON file in a directory."""
    input_files = iter_input_files(input_dir, pattern)
    errors: list[dict[str, str]] = []
    success_count = 0

    for index, input_file in enumerate(input_files, start=1):
        output_path = build_batch_output_path(input_file, input_dir, output_dir)
        try:
            parse_json_file(
                input_file,
                output_path,
                dialect=dialect,
                deduplicate=deduplicate,
            )
            success_count += 1
        except Exception as exc:
            errors.append(
                {
                    "file": str(input_file),
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )

        print(
            f"[{index}/{len(input_files)}] {input_file} -> {output_path}",
            file=sys.stderr,
        )

    return success_count, errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert eval_results correct/wrong SQL strings or legacy all_sqls "
            "JSON files to sqlglot AST trees."
        )
    )
    parser.add_argument("input", type=Path, help="Path to an input JSON file or directory.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output file for file input, or output directory for directory input. "
            "File input defaults to stdout; directory input defaults to <input>_ast."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for directory input. Defaults to *.json.",
    )
    parser.add_argument(
        "--dialect",
        help="Optional sqlglot source dialect, for example sqlite, mysql, or postgres.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help=(
            "Parse only unique SQL strings within each correct_set, wrong_set, "
            "or all_sqls list."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.input.is_dir():
        output_dir = args.output or args.input.with_name(f"{args.input.name}_ast")
        success_count, errors = parse_directory(
            args.input,
            output_dir,
            pattern=args.pattern,
            dialect=args.dialect,
            deduplicate=args.deduplicate,
        )

        print(
            json.dumps(
                {
                    "input_dir": str(args.input),
                    "output_dir": str(output_dir),
                    "success_count": success_count,
                    "error_count": len(errors),
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if errors else 0

    if not args.input.is_file():
        raise SystemExit(f"Input path does not exist or is not a file/directory: {args.input}")

    with args.input.open("r", encoding="utf-8") as file:
        data = json.load(file)

    parsed = parse_json_data(
        data,
        source_path=args.input,
        dialect=args.dialect,
        deduplicate=args.deduplicate,
    )
    if args.output:
        output = json.dumps(parsed, ensure_ascii=False, indent=2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
