"""Schema-linking penalty for predicted SQL.

Penalizes references to tables or columns that do not exist in the
database schema.  The penalty is clipped to [0, 1].
"""

from __future__ import annotations

from sqlglot import expressions as exp


def extract_referenced_tables(ast: exp.Expression) -> set[str]:
    """Return all table names (lowercase) referenced anywhere in the AST."""
    return {tbl.name.lower() for tbl in ast.find_all(exp.Table) if tbl.name}


def extract_referenced_columns(ast: exp.Expression) -> set[str]:
    """Return 'table.column' strings (lowercase) referenced in the AST.

    Columns without an explicit table qualifier use the prefix ``*``,
    e.g. ``*.id``.
    """
    result: set[str] = set()
    for col in ast.find_all(exp.Column):
        col_name = col.name.lower()
        if not col_name:
            continue
        table = col.table
        tbl_str = table.lower() if isinstance(table, str) and table else "*"
        result.add(f"{tbl_str}.{col_name}")
    return result


def schema_penalty(
    ast_pred: exp.Expression,
    valid_columns: set[str],
    valid_tables: set[str],
    col_penalty: float = 0.1,
    table_penalty: float = 0.2,
) -> float:
    """Compute schema-violation penalty, clipped to [0, 1].

    Parameters
    ----------
    ast_pred:
        AST of the predicted SQL.
    valid_columns:
        Set of ``'table.column'`` strings from the database schema.
    valid_tables:
        Set of table names from the database schema.
    col_penalty:
        Per-column penalty for each unqualified or unknown column reference.
    table_penalty:
        Per-table penalty for each unknown table reference.
    """
    pred_tables = extract_referenced_tables(ast_pred)
    pred_cols   = extract_referenced_columns(ast_pred)

    invalid_tables = pred_tables - valid_tables
    # Only penalise columns with an explicit table qualifier
    invalid_cols = {
        ref for ref in pred_cols
        if not ref.startswith("*.") and ref not in valid_columns
    }

    penalty = table_penalty * len(invalid_tables) + col_penalty * len(invalid_cols)
    return min(penalty, 1.0)
