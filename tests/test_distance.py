"""Unit tests for the distance pipeline.

Run with:  pytest tests/test_distance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlglot import expressions as exp
import sqlglot

from src.normalize import normalize_sql, parse_to_ast, qualify_sql
from src.distance.component_f1 import (
    COMPONENTS,
    component_f1_distance,
    component_f1_scores,
    extract_groupby_cols,
    extract_having_conditions,
    extract_join_tables,
    extract_limit,
    extract_orderby_cols,
    extract_select_cols,
    extract_where_conditions,
    token_f1,
)
from src.distance.ted import count_nodes, normalized_ted
from src.distance.schema_penalty import (
    extract_referenced_columns,
    extract_referenced_tables,
    schema_penalty,
)
from src.distance.composite import (
    DistanceWeights,
    hierarchical_distance,
    hierarchical_distance_with_detail,
)
from src.utils.schema import (
    ColumnInfo,
    DBSchema,
    TableSchema,
    get_column_set,
    get_table_set,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_schema() -> DBSchema:
    """Minimal in-memory DBSchema (no SQLite file needed)."""
    movies = TableSchema(
        name="movies",
        columns=[
            ColumnInfo("movie_id",       "INTEGER", True,  True),
            ColumnInfo("movie_title",    "TEXT",    False, True),
            ColumnInfo("release_year",   "INTEGER", False, False),
            ColumnInfo("movie_popularity","REAL",   False, False),
        ],
    )
    ratings = TableSchema(
        name="ratings",
        columns=[
            ColumnInfo("rating_id",    "INTEGER", True,  True),
            ColumnInfo("movie_id",     "INTEGER", False, True),
            ColumnInfo("rating_score", "REAL",    False, False),
            ColumnInfo("user_id",      "INTEGER", False, False),
        ],
    )
    return DBSchema(
        db_id="test_db",
        tables={"movies": movies, "ratings": ratings},
        table_names=["movies", "ratings"],
    )


@pytest.fixture
def schema() -> DBSchema:
    return _make_schema()


def _ast(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="sqlite")


# ── token_f1 ─────────────────────────────────────────────────────────────────

def test_token_f1_identical():
    assert token_f1(["a", "b"], ["a", "b"]) == 1.0


def test_token_f1_disjoint():
    assert token_f1(["a"], ["b"]) == 0.0


def test_token_f1_partial():
    score = token_f1(["a", "b"], ["b", "c"])
    assert 0.0 < score < 1.0


def test_token_f1_both_empty():
    assert token_f1([], []) == 1.0


def test_token_f1_pred_empty():
    assert token_f1([], ["a"]) == 0.0


def test_token_f1_gold_empty():
    assert token_f1(["a"], []) == 0.0


def test_token_f1_multiset():
    # Duplicate tokens count separately
    assert token_f1(["a", "a"], ["a"]) < 1.0


# ── component extraction ──────────────────────────────────────────────────────

def test_extract_select_cols_simple():
    ast = _ast("SELECT movie_title FROM movies")
    items = extract_select_cols(ast)
    assert len(items) == 1
    assert "(none,movie_title)" in items


def test_extract_select_cols_agg():
    ast = _ast("SELECT AVG(rating_score) FROM ratings")
    items = extract_select_cols(ast)
    assert any("avg" in item for item in items)


def test_extract_select_cols_star():
    ast = _ast("SELECT * FROM movies")
    items = extract_select_cols(ast)
    assert "(*,*)" in items


def test_extract_where_conditions_single():
    ast = _ast("SELECT 1 FROM movies WHERE release_year = 1945")
    conds = extract_where_conditions(ast)
    assert len(conds) == 1


def test_extract_where_conditions_and():
    ast = _ast("SELECT 1 FROM movies WHERE release_year = 1945 AND movie_title = 'x'")
    conds = extract_where_conditions(ast)
    assert len(conds) == 2


def test_extract_groupby_cols():
    ast = _ast("SELECT movie_id, COUNT(*) FROM ratings GROUP BY movie_id")
    cols = extract_groupby_cols(ast)
    assert "movie_id" in cols


def test_extract_orderby_cols():
    ast = _ast("SELECT movie_title FROM movies ORDER BY movie_popularity DESC")
    items = extract_orderby_cols(ast)
    assert any("desc" in item for item in items)


def test_extract_join_tables():
    ast = _ast("SELECT * FROM movies JOIN ratings ON movies.movie_id = ratings.movie_id")
    tables = extract_join_tables(ast)
    assert "movies" in tables
    assert "ratings" in tables


def test_extract_limit():
    ast = _ast("SELECT movie_title FROM movies LIMIT 10")
    assert extract_limit(ast) == ["10"]


def test_extract_limit_absent():
    ast = _ast("SELECT movie_title FROM movies")
    assert extract_limit(ast) == []


# ── component_f1_scores ───────────────────────────────────────────────────────

def test_component_f1_scores_all_keys():
    ast = _ast("SELECT movie_title FROM movies WHERE release_year = 1945")
    scores = component_f1_scores(ast, ast)
    assert set(scores.keys()) == set(COMPONENTS)


def test_component_f1_scores_identical():
    ast = _ast("SELECT movie_title FROM movies WHERE release_year = 1945")
    scores = component_f1_scores(ast, ast)
    for v in scores.values():
        assert v == 1.0


def test_component_f1_distance_identical():
    ast = _ast("SELECT movie_title FROM movies")
    assert component_f1_distance(ast, ast) == 0.0


def test_component_f1_distance_bounds():
    a = _ast("SELECT movie_title FROM movies WHERE release_year = 1945")
    b = _ast("SELECT rating_score FROM ratings GROUP BY movie_id")
    d = component_f1_distance(a, b)
    assert 0.0 <= d <= 1.0


def test_component_f1_distance_different_select():
    a = _ast("SELECT movie_title FROM movies")
    b = _ast("SELECT release_year FROM movies")
    d = component_f1_distance(a, b)
    assert d > 0.0


def test_component_f1_distance_custom_weights():
    a = _ast("SELECT movie_title FROM movies WHERE release_year = 1945")
    b = _ast("SELECT movie_title FROM movies WHERE release_year = 2000")
    w_default = component_f1_distance(a, b)
    w_custom  = component_f1_distance(a, b, weights={"select": 10.0, "where": 0.0,
                                                      "join": 0.0, "groupby": 0.0,
                                                      "orderby": 0.0, "having": 0.0,
                                                      "limit": 0.0})
    # With where weight = 0, only select matters → 0.0 (select is identical)
    assert w_custom == 0.0
    assert w_default > 0.0


# ── normalized_ted ────────────────────────────────────────────────────────────

def test_normalized_ted_identical():
    ast = _ast("SELECT movie_title FROM movies")
    assert normalized_ted(ast, ast) == 0.0


def test_normalized_ted_none_pred():
    ast = _ast("SELECT 1")
    assert normalized_ted(None, ast) == 1.0


def test_normalized_ted_none_gold():
    ast = _ast("SELECT 1")
    assert normalized_ted(ast, None) == 1.0


def test_normalized_ted_bounds():
    a = _ast("SELECT movie_title FROM movies WHERE release_year = 1945")
    b = _ast("SELECT COUNT(*) FROM ratings GROUP BY movie_id HAVING COUNT(*) > 5")
    d = normalized_ted(a, b)
    assert 0.0 <= d <= 1.0


def test_count_nodes():
    ast = _ast("SELECT 1")
    assert count_nodes(ast) >= 1


# ── schema_penalty ────────────────────────────────────────────────────────────

def test_extract_referenced_tables():
    ast = _ast("SELECT * FROM movies JOIN ratings ON movies.movie_id = ratings.movie_id")
    tables = extract_referenced_tables(ast)
    assert "movies" in tables
    assert "ratings" in tables


def test_extract_referenced_columns():
    ast = _ast("SELECT movies.movie_title FROM movies")
    cols = extract_referenced_columns(ast)
    assert "movies.movie_title" in cols


def test_schema_penalty_valid_sql(schema: DBSchema):
    ast = _ast("SELECT movies.movie_title FROM movies WHERE movies.release_year = 1945")
    valid_cols   = get_column_set(schema)
    valid_tables = get_table_set(schema)
    penalty = schema_penalty(ast, valid_cols, valid_tables)
    assert penalty == 0.0


def test_schema_penalty_invalid_table(schema: DBSchema):
    ast = _ast("SELECT * FROM nonexistent_table")
    valid_cols   = get_column_set(schema)
    valid_tables = get_table_set(schema)
    penalty = schema_penalty(ast, valid_cols, valid_tables)
    assert penalty > 0.0


def test_schema_penalty_clipped(schema: DBSchema):
    # Many invalid tables → penalty must not exceed 1.0
    ast = _ast("SELECT * FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.id = t3.id "
               "JOIN t4 ON t3.id = t4.id JOIN t5 ON t4.id = t5.id JOIN t6 ON t5.id = t6.id")
    valid_cols   = get_column_set(schema)
    valid_tables = get_table_set(schema)
    penalty = schema_penalty(ast, valid_cols, valid_tables)
    assert penalty <= 1.0


# ── hierarchical_distance ─────────────────────────────────────────────────────

GOLD = "SELECT movie_title FROM movies WHERE release_year = 1945 ORDER BY movie_popularity DESC"


def test_hierarchical_distance_identical(schema: DBSchema):
    d = hierarchical_distance(GOLD, GOLD, schema=schema)
    assert d == 0.0


def test_hierarchical_distance_unparsable(schema: DBSchema):
    d = hierarchical_distance("NOT VALID SQL !!!!", GOLD, schema=schema)
    assert d == 1.0


def test_hierarchical_distance_bounds(schema: DBSchema):
    pred = "SELECT movie_title FROM movies"
    d = hierarchical_distance(pred, GOLD, schema=schema)
    assert 0.0 <= d <= 1.0


def test_hierarchical_distance_ordering(schema: DBSchema):
    """A closer SQL should have a smaller distance."""
    close = "SELECT movie_title FROM movies WHERE release_year = 1945"
    far   = "SELECT COUNT(*) FROM ratings GROUP BY movie_id"
    d_close = hierarchical_distance(close, GOLD, schema=schema)
    d_far   = hierarchical_distance(far,   GOLD, schema=schema)
    assert d_close < d_far


def test_hierarchical_distance_custom_weights(schema: DBSchema):
    pred = "SELECT movie_title FROM movies"
    w1 = DistanceWeights(component_f1=0.5, ted=0.3, schema_penalty=0.2)
    w2 = DistanceWeights(component_f1=1.0, ted=0.0, schema_penalty=0.0)
    d1 = hierarchical_distance(pred, GOLD, schema=schema, weights=w1)
    d2 = hierarchical_distance(pred, GOLD, schema=schema, weights=w2)
    # Different weights → different distances (unless all sub-distances are equal)
    detail = hierarchical_distance_with_detail(pred, GOLD, schema=schema, weights=w1)
    if not (detail.component_f1 == detail.ted == detail.schema_penalty):
        assert d1 != d2


def test_hierarchical_distance_no_schema():
    """Schema=None should still work (penalty skipped)."""
    pred = "SELECT movie_title FROM movies WHERE release_year = 1945"
    d = hierarchical_distance(pred, GOLD)
    assert 0.0 <= d <= 1.0


def test_hierarchical_distance_detail_parse_failed(schema: DBSchema):
    detail = hierarchical_distance_with_detail("BAD SQL", GOLD, schema=schema)
    assert detail.parse_failed is True
    assert detail.composite == 1.0


# ── normalize helpers ─────────────────────────────────────────────────────────

def test_normalize_sql_returns_string():
    result = normalize_sql("SELECT * FROM movies")
    assert isinstance(result, str)
    assert len(result) > 0


def test_normalize_sql_bad_input():
    """normalize_sql must never raise."""
    result = normalize_sql("THIS IS NOT SQL AT ALL !!!")
    assert isinstance(result, str)


def test_parse_to_ast_valid():
    ast = parse_to_ast("SELECT 1")
    assert ast is not None


def test_parse_to_ast_invalid():
    ast = parse_to_ast("NOT SQL !!!")
    assert ast is None
