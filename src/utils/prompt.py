"""Format NL2SQL prompts for LLM training (SFT and DPO)."""

from __future__ import annotations


def format_schema_section(
    schema_dict: dict[str, list[str]],
    foreign_keys: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Format schema as a compact text block.

    Each table on one line: ``Table movies(id, title, year)``
    Foreign keys appended as ``FK: from_table.from_col -> to_table.to_col``
    """
    lines = []
    for table, cols in schema_dict.items():
        lines.append(f"Table {table}({', '.join(cols)})")
    if foreign_keys:
        for from_col, to_table, to_col, _ in foreign_keys:
            lines.append(f"FK: {from_col} -> {to_table}.{to_col}")
    return "\n".join(lines)


def format_nl2sql_prompt(
    question: str,
    schema_dict: dict[str, list[str]],
    evidence: str = "",
    foreign_keys: list[tuple[str, str, str, str]] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> str:
    """Build the full NL2SQL prompt string (no assistant reply).

    The returned string is used as the ``prompt`` field in DPO JSONL and
    as the prefix before the SQL completion in SFT data.
    """
    parts: list[str] = []

    if few_shot_examples:
        for ex in few_shot_examples:
            ex_schema = ex.get("schema", "")
            ex_q = ex.get("question", "")
            ex_ev = ex.get("evidence", "")
            ex_sql = ex.get("sql", "")
            block = f"### Example\nSchema:\n{ex_schema}\nQuestion: {ex_q}"
            if ex_ev:
                block += f"\nEvidence: {ex_ev}"
            block += f"\nSQL:\n{ex_sql}"
            parts.append(block)

    schema_str = format_schema_section(schema_dict, foreign_keys)
    parts.append(f"Schema:\n{schema_str}")
    parts.append(f"Question: {question}")
    if evidence and evidence.strip():
        parts.append(f"Evidence: {evidence.strip()}")
    parts.append("SQL:")

    return "\n\n".join(parts)


def format_sql_response(sql: str) -> str:
    """Return the assistant reply for a given SQL string."""
    return sql.strip()
