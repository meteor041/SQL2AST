"""Tree Edit Distance between two SQL ASTs.

Uses ``sqlglot.diff`` (an implementation of the GumTree algorithm) to count
the number of non-Keep edit operations, then normalizes by the size of the
larger tree.
"""

from __future__ import annotations

from sqlglot import expressions as exp
from sqlglot.diff import diff, Keep


def count_nodes(ast: exp.Expression) -> int:
    """Count all nodes in an AST by walking the entire tree."""
    return sum(1 for _ in ast.walk())


def normalized_ted(
    ast_pred: exp.Expression | None,
    ast_gold: exp.Expression | None,
) -> float:
    """Compute normalized TED = edit_ops / max(|tree_pred|, |tree_gold|).

    Returns 1.0 if either AST is None.  Clips result to [0, 1].
    Never raises.
    """
    if ast_pred is None or ast_gold is None:
        return 1.0
    try:
        edits = diff(ast_pred, ast_gold)
        edit_count = sum(1 for e in edits if not isinstance(e, Keep))
        denom = max(count_nodes(ast_pred), count_nodes(ast_gold))
        if denom == 0:
            return 0.0
        return min(edit_count / denom, 1.0)
    except Exception:
        return 1.0
