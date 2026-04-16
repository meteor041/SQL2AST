"""Hierarchical composite SQL distance — the single public entry point.

Formula
-------
D = w_comp * d_comp  +  w_ted * d_ted  +  w_sch * d_sch

where each sub-distance is in [0, 1] and the weights sum to 1.0.
An unparseable SQL immediately returns D = 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import expressions as exp

from src.normalize import normalize_sql, parse_to_ast
from src.distance.component_f1 import component_f1_distance
from src.distance.ted import normalized_ted
from src.distance.schema_penalty import schema_penalty as _schema_penalty


@dataclass
class DistanceWeights:
    """Sub-distance weights.  Should sum to 1.0."""
    component_f1:   float = 0.5
    ted:            float = 0.3
    schema_penalty: float = 0.2


@dataclass
class DistanceDetail:
    """Breakdown of each sub-distance, for debugging and calibration."""
    component_f1:   float
    ted:            float
    schema_penalty: float
    composite:      float
    parse_failed:   bool


def hierarchical_distance(
    sql_s: str,
    sql_g: str,
    schema: "DBSchema | None" = None,
    dialect: str = "sqlite",
    weights: DistanceWeights | None = None,
) -> float:
    """Compute D(sql_s, sql_g) ∈ [0, 1].

    Returns 1.0 if either SQL cannot be parsed.  Never raises.

    Parameters
    ----------
    sql_s:
        Sampled (predicted) SQL string.
    sql_g:
        Gold SQL string.
    schema:
        Optional ``DBSchema`` for schema-aware normalization and penalty.
        When *None* the schema penalty is skipped (set to 0.0).
    dialect:
        sqlglot dialect identifier, e.g. ``"sqlite"``.
    weights:
        Sub-distance weights.  Defaults to ``DistanceWeights()``.
    """
    return hierarchical_distance_with_detail(
        sql_s, sql_g, schema=schema, dialect=dialect, weights=weights
    ).composite


def hierarchical_distance_with_detail(
    sql_s: str,
    sql_g: str,
    schema: "DBSchema | None" = None,
    dialect: str = "sqlite",
    weights: DistanceWeights | None = None,
) -> DistanceDetail:
    """Same as ``hierarchical_distance`` but returns full ``DistanceDetail``."""
    w = weights or DistanceWeights()

    valid_tables:  set[str] = set()
    valid_columns: set[str] = set()
    schema_dict:   dict[str, list[str]] | None = None

    if schema is not None:
        from src.utils.schema import get_column_set, get_table_set, schema_to_prompt_dict
        valid_tables  = get_table_set(schema)
        valid_columns = get_column_set(schema)
        schema_dict   = schema_to_prompt_dict(schema)

    norm_s = normalize_sql(sql_s, schema=schema_dict, dialect=dialect)
    norm_g = normalize_sql(sql_g, schema=schema_dict, dialect=dialect)

    ast_s = parse_to_ast(norm_s, dialect=dialect)
    ast_g = parse_to_ast(norm_g, dialect=dialect)

    if ast_s is None or ast_g is None:
        return DistanceDetail(
            component_f1=1.0,
            ted=1.0,
            schema_penalty=1.0,
            composite=1.0,
            parse_failed=True,
        )

    d_comp = component_f1_distance(ast_s, ast_g)
    d_ted  = normalized_ted(ast_s, ast_g)
    d_sch  = (
        _schema_penalty(ast_s, valid_columns, valid_tables)
        if valid_tables else 0.0
    )

    composite = w.component_f1 * d_comp + w.ted * d_ted + w.schema_penalty * d_sch
    composite = max(0.0, min(1.0, composite))

    return DistanceDetail(
        component_f1=d_comp,
        ted=d_ted,
        schema_penalty=d_sch,
        composite=composite,
        parse_failed=False,
    )
