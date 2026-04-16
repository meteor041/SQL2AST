"""SQL semantic normalization using sqlglot.

Two public functions:
- ``normalize_sql``  → normalized SQL string (never raises)
- ``parse_to_ast``   → sqlglot Expression or None (never raises)

``qualify_sql`` is the internal step that runs qualify + simplify.
"""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

try:
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.optimizer.simplify import simplify as _simplify
    _HAS_OPTIMIZER = True
except ImportError:
    _HAS_OPTIMIZER = False


def _to_sqlglot_schema(schema: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """Convert {table: [col, ...]} to sqlglot optimizer format {table: {col: type}}."""
    return {tbl: {col: "TEXT" for col in cols} for tbl, cols in schema.items()}


def qualify_sql(
    sql: str,
    schema: dict[str, list[str]] | None = None,
    dialect: str = "sqlite",
) -> exp.Expression | None:
    """Parse and qualify SQL.  Returns None on any failure.

    Steps applied when available:
    1. ``sqlglot.parse_one``
    2. ``qualify`` (tables + columns if schema provided)
    3. ``simplify``
    """
    try:
        ast = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None

    if not _HAS_OPTIMIZER:
        return ast

    qualify_kwargs: dict = dict(
        dialect=dialect,
        qualify_tables=True,
        validate_qualify_columns=False,
        identify=False,
    )

    if schema:
        qualify_kwargs["schema"] = _to_sqlglot_schema(schema)
        qualify_kwargs["qualify_columns"] = True
    else:
        qualify_kwargs["qualify_columns"] = False

    try:
        ast = qualify(ast, **qualify_kwargs)
    except Exception:
        pass  # proceed with unqualified AST

    try:
        ast = _simplify(ast)
    except Exception:
        pass

    return ast


def normalize_sql(
    sql: str,
    schema: dict[str, list[str]] | None = None,
    dialect: str = "sqlite",
) -> str:
    """Return a normalized SQL string.  Never raises.

    Fallback chain:
    1. qualify_sql  → re-serialize
    2. parse_one    → re-serialize
    3. sql.strip()  (original string)
    """
    ast = qualify_sql(sql, schema=schema, dialect=dialect)
    if ast is not None:
        try:
            return ast.sql(dialect=dialect)
        except Exception:
            pass

    try:
        ast = sqlglot.parse_one(sql, read=dialect)
        return ast.sql(dialect=dialect)
    except Exception:
        pass

    return sql.strip()


def parse_to_ast(
    sql: str,
    dialect: str = "sqlite",
) -> exp.Expression | None:
    """Parse SQL to a live AST without qualify.  Returns None on failure.

    Used by distance modules after ``normalize_sql`` has already been applied.
    """
    try:
        return sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None
