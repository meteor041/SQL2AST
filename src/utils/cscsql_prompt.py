"""Build prompts by delegating to csc_sql's prompt construction helpers.

This keeps sql_rm aligned with the prompt format used by csc_sql inference:
- full ``CREATE TABLE`` DDL blocks
- sampled example values
- optional question-relevant DB values from Lucene retrieval
- evidence prepended to the question when present

The csc_sql module has heavy optional dependencies at import time (notably
Pyserini/Java), so imports are deferred until prompt construction is actually
requested.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


class CSCSQLPromptUnavailableError(RuntimeError):
    """Raised when csc_sql prompt helpers cannot be imported."""


def _default_cscsql_root() -> Path:
    return Path(__file__).resolve().parents[3] / "csc_sql"


def resolve_cscsql_root() -> Path:
    """Return the csc_sql repo root."""
    raw = os.environ.get("CSC_SQL_ROOT")
    return Path(raw).expanduser() if raw else _default_cscsql_root()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _load_process_dataset_module() -> ModuleType:
    """Import csc_sql's dataset prompt module lazily."""
    root = resolve_cscsql_root()
    src_path = root / "src"
    if not src_path.exists():
        raise CSCSQLPromptUnavailableError(
            f"CSC_SQL_ROOT does not contain src/: {src_path}. "
            "Set CSC_SQL_ROOT to the csc_sql repository root."
        )

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        return importlib.import_module("cscsql.service.process.process_dataset")
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise CSCSQLPromptUnavailableError(
            "Failed to import csc_sql prompt helpers. "
            "Make sure the csc_sql runtime dependencies are installed "
            "(including Java/javac for Pyserini) before running SFT/DPO prompt building. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _candidate_tables_json_paths(database_root: Path) -> list[Path]:
    stem = database_root.name
    candidates = [
        database_root / "train_tables.json",
        database_root / "tables.json",
    ]
    if stem.endswith("_databases"):
        candidates.append(database_root.parent / f"{stem[:-10]}_tables.json")
    return candidates


def _candidate_db_content_index_paths(database_root: Path) -> list[Path]:
    stem = database_root.name
    candidates: list[Path] = []
    env = os.environ.get("CSC_SQL_DB_CONTENT_INDEX_PATH")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            database_root.parent / "db_contents_index",
            database_root.parent / f"{stem}_contents_index",
            database_root.parent / f"{stem}_db_contents_index",
        ]
    )
    if stem.endswith("_databases"):
        prefix = stem[:-10]
        candidates.extend(
            [
                database_root.parent / f"{prefix}_contents_index",
                database_root.parent / f"{prefix}_db_contents_index",
            ]
        )
    return candidates


def resolve_tables_json_path(database_root: str | Path) -> Path:
    """Infer the schema-metadata JSON path associated with a DB root."""
    root = Path(database_root)
    for candidate in _candidate_tables_json_paths(root):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate tables metadata JSON for database root: {root}. "
        f"Tried: {', '.join(str(path) for path in _candidate_tables_json_paths(root))}"
    )


@lru_cache(maxsize=None)
def _load_db_info_map(tables_json_path_str: str) -> dict[str, dict[str, Any]]:
    records = json.loads(Path(tables_json_path_str).read_text(encoding="utf-8"))
    return {
        record["db_id"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("db_id"), str)
    }


def load_db_info(database_root: str | Path, db_id: str) -> dict[str, Any]:
    """Return the csc_sql schema metadata entry for a DB."""
    tables_json_path = resolve_tables_json_path(database_root)
    info_map = _load_db_info_map(str(tables_json_path))
    try:
        return info_map[db_id]
    except KeyError as exc:
        raise KeyError(
            f"db_id {db_id!r} not found in schema metadata: {tables_json_path}"
        ) from exc


def resolve_db_path(database_root: str | Path, db_id: str) -> Path:
    """Resolve nested or flat SQLite layout."""
    root = Path(database_root)
    nested = root / db_id / f"{db_id}.sqlite"
    if nested.exists():
        return nested
    flat = root / f"{db_id}.sqlite"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"SQLite DB not found: {nested} or {flat}")


def resolve_db_content_index_root(database_root: str | Path) -> Path | None:
    """Return the DB-content index root when available."""
    if not _env_flag("CSC_SQL_USE_RELEVANT_HITS", True):
        return None

    root = Path(database_root)
    for candidate in _candidate_db_content_index_paths(root):
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=None)
def _sampled_db_values(
    database_root_str: str,
    db_id: str,
    value_limit_num: int,
) -> dict[str, list[Any]]:
    module = _load_process_dataset_module()
    db_info = load_db_info(database_root_str, db_id)
    db_path = resolve_db_path(database_root_str, db_id)
    return module.sample_table_values(
        str(db_path),
        db_info["table_names_original"],
        value_limit_num,
    )


@lru_cache(maxsize=None)
def _lucene_searcher(index_root_str: str, db_id: str) -> Any:
    module = _load_process_dataset_module()
    searcher_path = Path(index_root_str) / db_id
    if not searcher_path.exists():
        raise FileNotFoundError(
            f"Lucene DB-content index for db_id {db_id!r} not found: {searcher_path}"
        )
    return module.LuceneSearcher(str(searcher_path))


@lru_cache(maxsize=None)
def _query_to_relevant_hits(
    index_root_str: str,
    db_id: str,
    question_text: str,
) -> dict[str, list[dict[str, Any]]]:
    module = _load_process_dataset_module()
    queries = module.obtain_n_grams(question_text, 8) + [question_text]
    queries = list(dict.fromkeys(queries))
    searcher = _lucene_searcher(index_root_str, db_id)
    return module.retrieve_relevant_hits(searcher, queries)


def _compose_question_text(question: str, evidence: str) -> str:
    if evidence.strip():
        return evidence + "\n" + question
    return question


def build_cscsql_prompt(
    *,
    question: str,
    db_id: str,
    database_root: str | Path,
    evidence: str = "",
    gold_sql: str = "",
    value_limit_num: int = 6,
    source: str = "ches",
    mode: str = "eval",
) -> str:
    """Return an input prompt built by csc_sql's prompt helpers.

    ``mode="eval"`` intentionally matches the full-schema prompt shape used by
    csc_sql inference, which is the target prompt distribution for SFT and DPO.
    """
    module = _load_process_dataset_module()
    db_root = str(Path(database_root))
    db_info = load_db_info(db_root, db_id)
    sampled_db_values_dict = _sampled_db_values(db_root, db_id, value_limit_num)
    db_id2relevant_hits = None
    index_root = resolve_db_content_index_root(db_root)
    if index_root is not None:
        try:
            question_text = _compose_question_text(question, evidence)
            db_id2relevant_hits = {
                db_id: _query_to_relevant_hits(str(index_root), db_id, question_text)
            }
        except FileNotFoundError:
            db_id2relevant_hits = None

    item = module.prepare_input_output_pairs(
        data={
            "question": question,
            "evidence": evidence,
            "db_id": db_id,
            "SQL": gold_sql,
        },
        ek_key="evidence",
        db_id2relevant_hits=db_id2relevant_hits,
        sampled_db_values_dict=sampled_db_values_dict,
        db_info=db_info,
        source=source,
        output_key="SQL",
        mode=mode,
    )
    return item["input_seq"]
