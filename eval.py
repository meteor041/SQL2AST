"""Evaluate sampled SQL clusters against BIRD gold SQL by execution result."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cscsql.utils.file_utils import FileUtils
from cscsql.utils.sqlite_db_utils import SqliteDbUtils
from tqdm import tqdm


SAMPLE_FILE_RE = re.compile(r"^(\d+)_.*\.json$")

# 输出类型定义
# rows: 查询结果的行数据，列表中的每个元素都是一个行数据列表
# row_count: 查询结果的行数
# error: 如果执行过程中发生错误，包含错误类型和消息的字典；如果没有错误，则为 None
@dataclass
class QueryResult:
    rows: list[list[Any]]
    row_count: int
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "row_count": self.row_count,
            "error": self.error,
        }

# 获取TRAIN_DATA_PATH、TRAIN_DATABASE_PATH和SQL_PATH的值，并进行基本的验证
def parse_location(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Location file not found: {path}")

    values: dict[str, str] = {}
    for line in FileUtils.read_to_text_list(str(path), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    missing = {"TRAIN_DATA_PATH", "TRAIN_DATABASE_PATH", "SQL_PATH"} - values.keys()
    if missing:
        raise ValueError(f"Missing required location keys: {sorted(missing)}")
    return values

# 获取训练数据的函数，读取指定路径的JSON文件，并验证其内容是否为列表
def load_train_data(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Train data file not found: {path}")

    data = FileUtils.load_json(str(path))
    if not isinstance(data, list):
        raise ValueError(f"Expected train data to be a list: {path}")
    return data

def is_sample_file(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.endswith("_ast.json")
        and SAMPLE_FILE_RE.match(path.name) is not None
    )


# 解析SQL采样文件
def discover_sample_files(sql_path: Path, limit: int | None = None) -> list[Path]:
    if sql_path.is_file():
        selected = [sql_path] if is_sample_file(sql_path) else []
        if limit is not None:
            selected = selected[:limit]
        return selected

    if not sql_path.is_dir():
        raise FileNotFoundError(f"SQL_PATH is not a file or directory: {sql_path}")

    files: list[tuple[int, Path]] = []
    for raw_path in FileUtils.list_file_prefix(
        str(sql_path), add_parent=True, end_with=".json"
    ):
        path = Path(raw_path)
        if not is_sample_file(path):
            continue
        match = SAMPLE_FILE_RE.match(path.name)
        if not match:
            continue
        files.append((int(match.group(1)), path))

    files.sort(key=lambda item: (item[0], item[1].name))
    selected = [path for _, path in files]
    if limit is not None:
        selected = selected[:limit]
    return selected


def resolve_output_dir(args: argparse.Namespace, location: dict[str, str]) -> Path:
    output_dir = (
        args.output_dir
        or args.output
        or location.get("EVAL_OUTPUT_PATH")
        or location.get("OUTPUT_PATH")
        or "eval_results"
    )
    return Path(output_dir)


def sample_id_from_path(path: Path) -> int:
    match = SAMPLE_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Input file name does not contain a sample id: {path}")
    return int(match.group(1))

# 获取SQL聚类结果，返回一个字典，键是标准化的SQL文本，值是包含SQL文本、出现次数和出现位置的字典
def load_sql_clusters(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    data = FileUtils.load_json(str(path))

    if not isinstance(data, list):
        raise ValueError(f"Expected sample file to contain a JSON list: {path}")

    clusters: dict[str, dict[str, Any]] = {}
    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        all_sqls = record.get("all_sqls", [])
        if not isinstance(all_sqls, list):
            raise ValueError(f"Expected all_sqls to be a list in {path}")

        for sql_index, sql in enumerate(all_sqls):
            if not isinstance(sql, str):
                continue
            normalized = sql.strip()
            if not normalized:
                continue
            # setdefault代表如果normalized已经在clusters中，则返回对应的value；如果没有，则创建一个新的entry，key为normalized，value为提供的默认字典，并返回这个新创建的字典
            cluster = clusters.setdefault(
                normalized,
                {
                    "sql": normalized,
                    "count": 0,
                    "occurrences": [],
                },
            )
            cluster["count"] += 1
            cluster["occurrences"].append(
                {
                    "record_index": record_index,
                    "sql_index": sql_index,
                }
            )

    return clusters

# 获取sqlite数据库的可能路径，首先是平铺路径（database_root/db_id.sqlite），其次是嵌套路径（database_root/db_id/db_id.sqlite）
def sqlite_path_candidates(database_root: Path, db_id: str) -> list[Path]:
    flat_path = database_root / f"{db_id}.sqlite"
    nested_path = database_root / db_id / f"{db_id}.sqlite"
    return [flat_path, nested_path]

# 检查sqlite数据库文件是否存在，优先使用平铺路径，如果平铺路径不存在则使用嵌套路径，如果两者都不存在则抛出FileNotFoundError
def resolve_sqlite_path(database_root: Path, db_id: str) -> Path:
    flat_path, nested_path = sqlite_path_candidates(database_root, db_id)
    if flat_path.exists():
        return flat_path

    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(
        f"SQLite database not found. Tried {flat_path} and {nested_path}"
    )


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    return value


def rows_to_json(rows: list[tuple[Any, ...]]) -> list[list[Any]]:
    return [[json_value(value) for value in row] for row in rows]

# 执行SQL查询，并返回查询结果或错误信息
def execute_sql(sqlite_path: Path, sql: str, timeout: float) -> QueryResult:
    try:
        rows = SqliteDbUtils.execute_sql(str(sqlite_path), sql, meta_time_out=timeout)
        if rows == [("timeout",)]:
            return QueryResult(
                rows=[],
                row_count=0,
                error={
                    "type": "TimeoutError",
                    "message": f"SQL execution exceeded {timeout} seconds",
                },
            )
        return QueryResult(rows=rows_to_json(rows), row_count=len(rows))
    except Exception as exc:
        return QueryResult(
            rows=[],
            row_count=0,
            error={
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        )


def row_key(row: list[Any]) -> str:
    return FileUtils.dump_json_string(row, sort_keys=True)


def results_equal(
    gold_result: QueryResult,
    pred_result: QueryResult,
    ignore_order: bool = False,
) -> bool:
    if gold_result.error or pred_result.error:
        return False
    if ignore_order:
        return Counter(row_key(row) for row in gold_result.rows) == Counter(
            row_key(row) for row in pred_result.rows
        )
    return gold_result.rows == pred_result.rows


def execute_cluster_model(
    cluster_index: int,
    cluster: dict[str, Any],
    sqlite_path: Path,
    timeout: float,
) -> dict[str, Any]:
    pred_result = execute_sql(sqlite_path, cluster["sql"], timeout)
    return {
        "cluster_index": cluster_index,
        "cluster": cluster,
        "pred_result": pred_result,
    }


def sort_cluster_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: item["cluster_index"])


def run_clusters_parallel(
    clusters: dict[str, dict[str, Any]],
    sqlite_path: Path,
    timeout: float,
    num_cpus: int,
) -> list[dict[str, Any]]:
    cluster_items = list(clusters.values())
    if num_cpus <= 1:
        return [
            execute_cluster_model(
                cluster_index=cluster_index,
                cluster=cluster,
                sqlite_path=sqlite_path,
                timeout=timeout,
            )
            for cluster_index, cluster in enumerate(cluster_items)
        ]

    results: list[dict[str, Any]] = []

    def result_callback(result: dict[str, Any]) -> None:
        results.append(result)

    pool = mp.Pool(processes=num_cpus)
    for cluster_index, cluster in enumerate(cluster_items):
        pool.apply_async(
            execute_cluster_model,
            args=(cluster_index, cluster, sqlite_path, timeout),
            callback=result_callback,
        )
    pool.close()
    pool.join()

    return sort_cluster_results(results)


def make_cluster_record(
    sample_id: int,
    file_path: Path,
    db_id: str,
    gold_sql: str,
    cluster: dict[str, Any],
    gold_result: QueryResult,
    pred_result: QueryResult,
    is_correct: bool,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "file": str(file_path),
        "db_id": db_id,
        "gold_sql": gold_sql,
        "sql": cluster["sql"],
        "count": cluster["count"],
        "occurrences": cluster["occurrences"],
        "gold_result": gold_result.to_dict(),
        "pred_result": pred_result.to_dict(),
        "is_correct": is_correct,
    }


def make_output_path(output_dir: Path, file_path: Path) -> Path:
    return output_dir / f"{FileUtils.get_file_name(str(file_path))}_eval.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    if not FileUtils.dump_json(str(path), data, indent=2):
        raise OSError(f"Failed to write JSON file: {path}")


def evaluate_file(
    file_path: Path,
    train_data: list[dict[str, Any]],
    database_root: Path,
    timeout: float,
    ignore_order: bool,
    num_cpus: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    sample_id = sample_id_from_path(file_path)
    if sample_id >= len(train_data):
        return [], [], {
            "file": str(file_path),
            "sample_id": sample_id,
            "type": "IndexError",
            "message": f"sample_id {sample_id} is outside train data length {len(train_data)}",
        }

    train_item = train_data[sample_id]
    db_id = train_item.get("db_id")
    gold_sql = train_item.get("SQL")
    if not isinstance(db_id, str) or not isinstance(gold_sql, str):
        return [], [], {
            "file": str(file_path),
            "sample_id": sample_id,
            "type": "ValueError",
            "message": "Train item must contain string db_id and SQL fields",
        }

    try:
        sqlite_path = resolve_sqlite_path(database_root, db_id)
        clusters = load_sql_clusters(file_path)
    except Exception as exc:
        return [], [], {
            "file": str(file_path),
            "sample_id": sample_id,
            "db_id": db_id,
            "type": exc.__class__.__name__,
            "message": str(exc),
        }

    gold_result = execute_sql(sqlite_path, gold_sql, timeout)
    if gold_result.error:
        for candidate_path in sqlite_path_candidates(database_root, db_id):
            if candidate_path == sqlite_path or not candidate_path.exists():
                continue
            fallback_gold_result = execute_sql(candidate_path, gold_sql, timeout)
            if not fallback_gold_result.error:
                sqlite_path = candidate_path
                gold_result = fallback_gold_result
                break

    if gold_result.error:
        return [], [], {
            "file": str(file_path),
            "sample_id": sample_id,
            "db_id": db_id,
            "sqlite_path": str(sqlite_path),
            "gold_sql": gold_sql,
            "gold_result": gold_result.to_dict(),
            "type": "GoldExecutionError",
            "message": gold_result.error["message"],
        }

    correct_records: list[dict[str, Any]] = []
    wrong_records: list[dict[str, Any]] = []
    cluster_results = run_clusters_parallel(
        clusters=clusters,
        sqlite_path=sqlite_path,
        timeout=timeout,
        num_cpus=num_cpus,
    )
    for cluster_result in cluster_results:
        cluster = cluster_result["cluster"]
        pred_result = cluster_result["pred_result"]
        is_correct = results_equal(gold_result, pred_result, ignore_order=ignore_order)
        record = make_cluster_record(
            sample_id=sample_id,
            file_path=file_path,
            db_id=db_id,
            gold_sql=gold_sql,
            cluster=cluster,
            gold_result=gold_result,
            pred_result=pred_result,
            is_correct=is_correct,
        )
        if is_correct:
            correct_records.append(record)
        else:
            wrong_records.append(record)

    return correct_records, wrong_records, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster sampled SQL by text and evaluate by SQLite execution."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override SQL_PATH from .location with a sampled SQL file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override EVAL_OUTPUT_PATH/OUTPUT_PATH from .location.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Backward-compatible alias for --output-dir.",
    )
    parser.add_argument("--location", type=Path, default=Path(".location"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ignore-order",
        action="store_true",
        help="Compare result rows as multisets instead of strict row order.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    location = parse_location(args.location)
    train_data_path = Path(location["TRAIN_DATA_PATH"])
    database_root = Path(location["TRAIN_DATABASE_PATH"])
    sql_path = args.input_dir or Path(location["SQL_PATH"])
    output_dir = resolve_output_dir(args, location)
    train_data = load_train_data(train_data_path)
    sample_files = discover_sample_files(sql_path, limit=args.limit)

    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"Output path exists and is not a directory: {output_dir}")

    total_correct_count = 0
    total_wrong_count = 0
    file_errors: list[dict[str, Any]] = []
    output_files: list[str] = []

    with tqdm(
        sample_files,
        total=len(sample_files),
        desc="Evaluating samples",
        unit="file",
        file=sys.stderr,
    ) as progress:
        for file_path in progress:
            sample_id = sample_id_from_path(file_path)
            progress.set_postfix(sample_id=sample_id, refresh=False)

            correct_records, wrong_records, file_error = evaluate_file(
                file_path=file_path,
                train_data=train_data,
                database_root=database_root,
                timeout=args.timeout,
                ignore_order=args.ignore_order,
                num_cpus=args.num_cpus,
            )
            total_correct_count += len(correct_records)
            total_wrong_count += len(wrong_records)
            if file_error is not None:
                file_errors.append(file_error)

            sample_result = {
                "metadata": {
                    "sample_file": str(file_path),
                    "sample_id": sample_id,
                    "sql_path": str(sql_path),
                    "output_dir": str(output_dir),
                    "location": str(args.location),
                    "train_data_path": str(train_data_path),
                    "train_database_path": str(database_root),
                    "timeout": args.timeout,
                    "num_cpus": args.num_cpus,
                    "compare_mode": "ignore_order" if args.ignore_order else "strict_order",
                    "correct_count": len(correct_records),
                    "wrong_count": len(wrong_records),
                    "file_error_count": 1 if file_error is not None else 0,
                },
                "correct_set": correct_records,
                "wrong_set": wrong_records,
                "file_errors": [file_error] if file_error is not None else [],
            }

            output_path = make_output_path(output_dir, file_path)
            write_json(output_path, sample_result)
            output_files.append(str(output_path))
            progress.set_postfix(
                sample_id=sample_id,
                correct=total_correct_count,
                wrong=total_wrong_count,
                errors=len(file_errors),
                refresh=False,
            )

    summary = {
        "metadata": {
            "sql_path": str(sql_path),
            "output_dir": str(output_dir),
            "location": str(args.location),
            "train_data_path": str(train_data_path),
            "train_database_path": str(database_root),
            "timeout": args.timeout,
            "num_cpus": args.num_cpus,
            "compare_mode": "ignore_order" if args.ignore_order else "strict_order",
            "sample_file_count": len(sample_files),
            "correct_count": total_correct_count,
            "wrong_count": total_wrong_count,
            "file_error_count": len(file_errors),
        },
        "file_errors": file_errors,
        "output_files": output_files,
    }

    write_json(output_dir / "summary.json", summary)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
                "sample_file_count": len(sample_files),
                "correct_count": total_correct_count,
                "wrong_count": total_wrong_count,
                "file_error_count": len(file_errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if file_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
