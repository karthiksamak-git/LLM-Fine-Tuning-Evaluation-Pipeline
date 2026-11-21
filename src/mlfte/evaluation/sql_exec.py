"""Execution-match harness for text-to-SQL.

For each item we build an in-memory SQLite database from the example's ``CREATE TABLE``
statements, fill it with *deterministic synthetic rows*, then run the gold query and the
prediction on the same database and compare result multisets.

Row generation is type-aware and seeded from the schema, so the same example always gets
the same database. Literals appearing in the gold query are injected into the value pools
so that ``WHERE`` clauses can actually match rows.

Limitations (also reported in every metrics file):

* Synthetic rows are not real data. Two different queries can agree on a toy database
  while differing on real data (false equivalences), and a semantically correct query can
  still disagree with the gold query if the gold query depends on data distribution.
* Tables are small (``rows_per_table``), so aggregates collapse to a few values.
* SQLite semantics are assumed; other dialects (e.g. ``DATE_TRUNC``) fail validity.
* Only read-only statements are executed; anything else is scored invalid.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

READ_ONLY_ROOTS = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Subquery,
    exp.With,
)
_NUMERIC_TYPES = ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE", "SERIAL")
_TEXT_POOL = ["alpha", "beta", "gamma", "delta", "epsilon"]
_DATE_POOL = ["2020-01-01", "2021-06-15", "2022-12-31"]
_LIMITATIONS = (
    "Execution match runs on synthetic, seeded toy rows (6 per table by default), not the "
    "original databases: false equivalences and false mismatches are both possible.",
    "Aggregations and JOINs over the synthetic rows are only weakly discriminative.",
    "SQLite is the only dialect executed; dialect-specific SQL is reported as invalid.",
)


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple[Any, ...]] | None = None
    columns: list[str] | None = None
    error: str | None = None
    reason: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class ExecutionComparison:
    is_valid: bool
    execution_match: bool
    gold_ok: bool
    gold_error: str | None
    pred_error: str | None
    pred_reason: str | None
    gold_row_count: int | None
    pred_row_count: int | None
    order_sensitive: bool
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _column_category(type_name: str) -> str:
    t = (type_name or "").upper()
    if any(k in t for k in _NUMERIC_TYPES):
        return "numeric"
    if "DATE" in t or "TIME" in t:
        return "date"
    if "BOOL" in t:
        return "numeric"
    return "text"


def parse_schema(schema_sql: str) -> dict[str, dict[str, str]]:
    """Parse CREATE TABLE statements into ``{table: {column: type}}``."""
    tables: dict[str, dict[str, str]] = {}
    try:
        statements = sqlglot.parse(schema_sql, dialect="sqlite")
    except Exception:
        statements = []
    for stmt in statements:
        if stmt is None or not isinstance(stmt, exp.Create):
            continue
        schema = stmt.this
        if not isinstance(schema, exp.Schema):
            continue
        table_expr = schema.this
        table_name = table_expr.name if hasattr(table_expr, "name") else str(table_expr)
        cols: dict[str, str] = {}
        for col in schema.expressions:
            if isinstance(col, exp.ColumnDef):
                kind = col.args.get("kind")
                kind_sql = kind.sql(dialect="sqlite") if kind is not None else "TEXT"
                cols[col.name] = kind_sql
        if cols:
            tables[table_name] = cols
    if tables:
        return tables
    # Fallback: tolerate unusual DDL that sqlglot rejects.
    for match in re.finditer(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\((.*?)\)\s*;",
        schema_sql,
        re.IGNORECASE | re.DOTALL,
    ):
        name, body = match.group(1), match.group(2)
        cols = {}
        for part in body.split(","):
            bits = part.strip().split()
            if len(bits) >= 2 and bits[0].upper() not in ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK"):
                cols[bits[0].strip('`"[]')] = " ".join(bits[1:])
        if cols:
            tables[name] = cols
    return tables


def gold_literals(gold_sql: str) -> tuple[list[Any], list[str]]:
    """Numeric and string literals used by the gold query (fed into synthetic rows)."""
    numbers: list[Any] = []
    texts: list[str] = []
    try:
        tree = sqlglot.parse_one(gold_sql, dialect="sqlite")
        for literal in tree.find_all(exp.Literal):
            if literal.is_string:
                if literal.this not in texts:
                    texts.append(str(literal.this))
            else:
                try:
                    value = float(literal.this)
                    numbers.append(int(value) if value.is_integer() else value)
                except (TypeError, ValueError):
                    continue
    except Exception:
        for match in re.finditer(r"'([^']*)'", gold_sql):
            texts.append(match.group(1))
        for match in re.finditer(r"(?<![\w.])(\d+)(?![\w.])", gold_sql):
            numbers.append(int(match.group(1)))
    return list(dict.fromkeys(numbers)), list(dict.fromkeys(texts))


def _numeric_pool(literals: list[Any]) -> list[Any]:
    pool: list[Any] = []
    for value in literals:
        if value not in pool:
            pool.append(value)
        # Neighbours make comparison operators discriminative.
        for delta in (-1, 1):
            neighbour = (value + delta) if isinstance(value, float) else value + delta
            if neighbour not in pool:
                pool.append(neighbour)
    for value in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        if value not in pool:
            pool.append(value)
    return pool


def _text_pool(literals: list[str]) -> list[str]:
    return list(dict.fromkeys([*literals, *_TEXT_POOL]))


def build_database(
    schema_sql: str,
    gold_sql: str,
    seed: int = 0,
    rows_per_table: int = 6,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Create an in-memory SQLite DB populated with deterministic synthetic rows."""
    tables = parse_schema(schema_sql)
    conn = sqlite3.connect(":memory:")
    info: dict[str, Any] = {"tables": {}, "rows_per_table": rows_per_table, "seed": seed}
    if not tables:
        info["error"] = "no CREATE TABLE statements could be parsed"
        return conn, info

    numbers, texts = gold_literals(gold_sql)
    num_pool = _numeric_pool(numbers)
    txt_pool = _text_pool(texts)

    seed_material = hashlib.sha256(f"{schema_sql}|{seed}".encode()).hexdigest()
    base_seed = int(seed_material[:8], 16)

    for table, columns in tables.items():
        cols_sql = ", ".join(f'"{name}"' for name in columns)
        conn.execute(f'CREATE TABLE "{table}" ({cols_sql})')
        rows: list[tuple[Any, ...]] = []
        for row_index in range(rows_per_table):
            values: list[Any] = []
            for col_index, (_col, col_type) in enumerate(columns.items()):
                digest = hashlib.sha256(
                    f"{base_seed}:{table}:{row_index}:{col_index}".encode()
                ).hexdigest()
                pick = int(digest[:8], 16)
                category = _column_category(col_type)
                if category == "numeric":
                    values.append(num_pool[pick % len(num_pool)])
                elif category == "date":
                    values.append(_DATE_POOL[pick % len(_DATE_POOL)])
                else:
                    values.append(txt_pool[pick % len(txt_pool)])
            rows.append(tuple(values))
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
        info["tables"][table] = {
            "columns": list(columns),
            "row_count": len(rows),
            "rows": [list(r) for r in rows],
        }
    conn.commit()
    return conn, info


def _is_read_only(sql: str) -> tuple[bool, str | None]:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="sqlite") if s is not None]
    except Exception as exc:
        return False, f"parse_error: {exc}"
    if not statements:
        return False, "empty_statement"
    if len(statements) > 1:
        return False, "multiple_statements"
    root = statements[0]
    node = root
    while isinstance(node, exp.Subquery) and isinstance(node.this, exp.Expression):
        node = node.this
    if not isinstance(node, READ_ONLY_ROOTS):
        return False, f"not_read_only: {type(root).__name__}"
    return True, None


def _install_timeout(conn: sqlite3.Connection, deadline: float) -> None:
    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, 2000)


def execute_sql(
    conn: sqlite3.Connection, sql: str, timeout_s: float = 2.0, max_rows: int = 5000
) -> ExecutionResult:
    """Execute one read-only statement with a wall-clock timeout."""
    if not sql or not sql.strip():
        return ExecutionResult(ok=False, error="empty_sql", reason="empty_sql")
    read_only, reason = _is_read_only(sql)
    if not read_only:
        return ExecutionResult(ok=False, reason=reason, error=reason)

    deadline = time.monotonic() + float(timeout_s)
    _install_timeout(conn, deadline)
    start = time.perf_counter()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows)
        columns = [d[0] for d in cursor.description or []]
        elapsed = (time.perf_counter() - start) * 1000.0
        return ExecutionResult(
            ok=True,
            rows=[tuple(normalize_value(v) for v in row) for row in rows],
            columns=columns,
            elapsed_ms=round(elapsed, 3),
        )
    except sqlite3.OperationalError as exc:
        message = str(exc)
        reason = "timeout" if "interrupted" in message.lower() else "sqlite_error"
        return ExecutionResult(ok=False, error=message, reason=reason)
    except sqlite3.Error as exc:
        return ExecutionResult(ok=False, error=str(exc), reason="sqlite_error")
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(ok=False, error=str(exc), reason="unexpected_error")
    finally:
        conn.set_progress_handler(None, 0)


def has_order_by(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        return any(True for _ in tree.find_all(exp.Order))
    except Exception:
        return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))


def compare_results(
    gold: ExecutionResult, pred: ExecutionResult, order_sensitive: bool
) -> bool:
    if not gold.ok or not pred.ok:
        return False
    gold_rows = gold.rows or []
    pred_rows = pred.rows or []
    if len(gold_rows) != len(pred_rows):
        return False
    if order_sensitive:
        return gold_rows == pred_rows
    return sorted(gold_rows, key=repr) == sorted(pred_rows, key=repr)


def execution_match(
    schema_sql: str,
    gold_sql: str,
    pred_sql: str,
    seed: int = 0,
    timeout_s: float = 2.0,
    rows_per_table: int = 6,
) -> ExecutionComparison:
    """Score one prediction against the gold query on a shared synthetic database."""
    conn, info = build_database(schema_sql, gold_sql, seed=seed, rows_per_table=rows_per_table)
    try:
        gold = execute_sql(conn, gold_sql, timeout_s)
        pred = execute_sql(conn, pred_sql, timeout_s)
    finally:
        conn.close()

    order_sensitive = has_order_by(gold_sql)
    match = compare_results(gold, pred, order_sensitive)
    if not pred.ok:
        reason = pred.reason
    elif not gold.ok:
        reason = "gold_execution_failed"
    elif match:
        reason = "match"
    else:
        reason = "result_mismatch"
    return ExecutionComparison(
        is_valid=bool(pred.ok),
        execution_match=bool(match),
        gold_ok=bool(gold.ok),
        gold_error=gold.error,
        pred_error=pred.error,
        pred_reason=pred.reason,
        gold_row_count=len(gold.rows) if gold.rows is not None else None,
        pred_row_count=len(pred.rows) if pred.rows is not None else None,
        order_sensitive=order_sensitive,
        reason=reason,
        extra={
            "synthetic_tables": info.get("tables", {}),
            "gold_columns": gold.columns,
            "pred_columns": pred.columns,
            "limitations": list(_LIMITATIONS),
        },
    )


def limitations() -> list[str]:
    return list(_LIMITATIONS)
