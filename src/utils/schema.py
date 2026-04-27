"""Load and represent SQLite database schema."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ColumnInfo:
    name: str       # lowercase
    col_type: str   # e.g. "TEXT", "INTEGER"
    is_pk: bool
    not_null: bool


@dataclass
class TableSchema:
    name: str                   # original case
    columns: list[ColumnInfo]
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)
    # (from_col, to_table, to_col, constraint_name)  — all lowercase


@dataclass
class DBSchema:
    db_id: str
    tables: dict[str, TableSchema]  # key: lowercase table name
    table_names: list[str] = field(default_factory=list)  # original-case, insertion order


def load_schema(db_path: str | Path) -> DBSchema:
    """Return DBSchema by reading PRAGMA metadata from a SQLite file."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    db_id = db_path.stem
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        raw_tables = [row[0] for row in cursor.fetchall()]

        tables: dict[str, TableSchema] = {}
        table_names: list[str] = []

        for tbl in raw_tables:
            table_names.append(tbl)

            cursor.execute(f"PRAGMA table_info([{tbl}])")
            cols = [
                ColumnInfo(
                    name=row[1].lower(),
                    col_type=(row[2] or "TEXT").upper(),
                    is_pk=bool(row[5]),
                    not_null=bool(row[3]),
                )
                for row in cursor.fetchall()
            ]

            cursor.execute(f"PRAGMA foreign_key_list([{tbl}])")
            fks = []
            for row in cursor.fetchall():
                from_col, to_table, to_col = row[3], row[2], row[4]
                if not from_col or not to_table or not to_col:
                    continue
                fks.append((from_col.lower(), to_table.lower(), to_col.lower(), ""))

            tables[tbl.lower()] = TableSchema(name=tbl, columns=cols, foreign_keys=fks)

        return DBSchema(db_id=db_id, tables=tables, table_names=table_names)
    finally:
        conn.close()


def get_column_set(schema: DBSchema) -> set[str]:
    """Return all 'table.column' strings (lowercase)."""
    result: set[str] = set()
    for tbl_name, tbl in schema.tables.items():
        for col in tbl.columns:
            result.add(f"{tbl_name}.{col.name}")
    return result


def get_table_set(schema: DBSchema) -> set[str]:
    """Return all table names (lowercase)."""
    return set(schema.tables.keys())


def schema_to_sqlglot_dict(schema: DBSchema) -> dict[str, dict[str, str]]:
    """Convert to sqlglot optimizer format: {table: {col: dtype}}."""
    return {
        tbl_name: {col.name: col.col_type for col in tbl.columns}
        for tbl_name, tbl in schema.tables.items()
    }


def schema_to_prompt_dict(schema: DBSchema) -> dict[str, list[str]]:
    """Return {table_name: [col_name, ...]} for prompt formatting."""
    return {
        tbl.name: [col.name for col in tbl.columns]
        for tbl in schema.tables.values()
    }
