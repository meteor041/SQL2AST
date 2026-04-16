"""Component-level F1 distance between two SQL ASTs.

The SQL is decomposed into 7 components.  Each component is compared as a
multiset and scored with token F1.  The final distance is
``1 - weighted_avg(F1)``.
"""

from __future__ import annotations

from collections import Counter

from sqlglot import expressions as exp

COMPONENTS = ("select", "where", "groupby", "orderby", "having", "limit", "join")

_DEFAULT_WEIGHTS: dict[str, float] = {
    "select":  2.0,
    "where":   2.0,
    "join":    2.0,
    "groupby": 1.0,
    "orderby": 1.0,
    "having":  1.0,
    "limit":   0.5,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _split_and(expr: exp.Expression) -> list[exp.Expression]:
    """Recursively flatten AND-connected predicates into a list."""
    if isinstance(expr, exp.And):
        return _split_and(expr.left) + _split_and(expr.right)
    return [expr]


def _agg_label(expr: exp.Expression) -> str:
    """Return a lowercase aggregate function name, or 'none'."""
    _map = {
        exp.Count: "count",
        exp.Sum:   "sum",
        exp.Avg:   "avg",
        exp.Max:   "max",
        exp.Min:   "min",
    }
    for cls, label in _map.items():
        if isinstance(expr, cls):
            return label
    if isinstance(expr, exp.AggFunc):
        return type(expr).__name__.lower()
    return "none"


# ── per-component extractors ──────────────────────────────────────────────────

def extract_select_cols(ast: exp.Expression) -> list[str]:
    """Extract SELECT items as canonical '(agg,col)' strings."""
    select = ast.find(exp.Select)
    if not select:
        return []
    items: list[str] = []
    for expr in select.expressions:
        inner = expr.this if isinstance(expr, exp.Alias) else expr
        if isinstance(inner, exp.Star):
            items.append("(*,*)")
        elif isinstance(inner, exp.AggFunc):
            agg = _agg_label(inner)
            col = inner.find(exp.Column)
            col_str = col.name.lower() if col else inner.sql().lower()
            items.append(f"({agg},{col_str})")
        elif isinstance(inner, exp.Column):
            items.append(f"(none,{inner.name.lower()})")
        else:
            items.append(f"(expr,{inner.sql().lower()})")
    return items


def extract_where_conditions(ast: exp.Expression) -> list[str]:
    """Extract individual WHERE predicates as SQL strings (lowercased)."""
    where = ast.find(exp.Where)
    if not where:
        return []
    return [c.sql().lower() for c in _split_and(where.this)]


def extract_groupby_cols(ast: exp.Expression) -> list[str]:
    group = ast.find(exp.Group)
    if not group:
        return []
    items: list[str] = []
    for expr in group.expressions:
        col = expr.find(exp.Column)
        items.append(col.name.lower() if col else expr.sql().lower())
    return items


def extract_orderby_cols(ast: exp.Expression) -> list[str]:
    order = ast.find(exp.Order)
    if not order:
        return []
    items: list[str] = []
    for ordered in order.expressions:
        col = ordered.find(exp.Column)
        col_str = col.name.lower() if col else ordered.this.sql().lower()
        direction = "desc" if ordered.args.get("desc") else "asc"
        items.append(f"({col_str},{direction})")
    return items


def extract_having_conditions(ast: exp.Expression) -> list[str]:
    having = ast.find(exp.Having)
    if not having:
        return []
    return [c.sql().lower() for c in _split_and(having.this)]


def extract_limit(ast: exp.Expression) -> list[str]:
    limit = ast.find(exp.Limit)
    if not limit:
        return []
    return [limit.this.sql().lower()]


def extract_join_tables(ast: exp.Expression) -> list[str]:
    """Collect table names from FROM + all JOIN clauses."""
    tables: list[str] = []
    from_node = ast.find(exp.From)
    if from_node:
        tbl = from_node.find(exp.Table)
        if tbl:
            tables.append(tbl.name.lower())
    for join in ast.find_all(exp.Join):
        tbl = join.find(exp.Table)
        if tbl:
            tables.append(tbl.name.lower())
    return tables


# ── F1 scoring ────────────────────────────────────────────────────────────────

def token_f1(pred: list[str], gold: list[str]) -> float:
    """Compute token-level F1 (multiset matching).  Both empty → 1.0."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred_c = Counter(pred)
    gold_c = Counter(gold)
    common = sum((pred_c & gold_c).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall    = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def component_f1_scores(
    ast_pred: exp.Expression,
    ast_gold: exp.Expression,
) -> dict[str, float]:
    """Return per-component F1 for all 7 COMPONENTS."""
    extractors = {
        "select":  extract_select_cols,
        "where":   extract_where_conditions,
        "groupby": extract_groupby_cols,
        "orderby": extract_orderby_cols,
        "having":  extract_having_conditions,
        "limit":   extract_limit,
        "join":    extract_join_tables,
    }
    return {
        name: token_f1(fn(ast_pred), fn(ast_gold))
        for name, fn in extractors.items()
    }


def component_f1_distance(
    ast_pred: exp.Expression,
    ast_gold: exp.Expression,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted component F1 distance = 1 - weighted_avg(F1).  Range [0, 1]."""
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    scores = component_f1_scores(ast_pred, ast_gold)

    total_w  = sum(w.get(k, 1.0) for k in COMPONENTS)
    weighted = sum(w.get(k, 1.0) * scores[k] for k in COMPONENTS)

    if total_w == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - weighted / total_w))
