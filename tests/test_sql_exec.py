"""Harness self-validation: gold-vs-gold, equivalent rewrites, wrong queries, malformed SQL."""

from __future__ import annotations

from mlfte.evaluation.sql_exec import (
    build_database,
    execute_sql,
    execution_match,
    has_order_by,
    parse_schema,
)

SCHEMA = (
    "CREATE TABLE head (head_id INTEGER, name TEXT, age INTEGER, born_state TEXT); "
    "CREATE TABLE department (department_id INTEGER, name TEXT, head_id INTEGER)"
)

GOLD_CASES = [
    "SELECT COUNT(*) FROM head WHERE age > 56",
    "SELECT name FROM head WHERE born_state = 'Alabama'",
    "SELECT head.name, department.name FROM head JOIN department ON head.head_id = department.head_id",
    "SELECT department.name FROM department WHERE department_id IN (SELECT head_id FROM head)",
    "SELECT AVG(age) FROM head",
    "SELECT name FROM head ORDER BY age DESC LIMIT 3",
    "SELECT COUNT(DISTINCT born_state) FROM head",
]

EQUIVALENT = [
    ("SELECT COUNT(*) FROM head WHERE age > 56", "select count( 1 ) from head where 56 < age;"),
    (
        "SELECT name FROM head WHERE born_state = 'Alabama'",
        "SELECT T1.name FROM head AS T1 WHERE T1.born_state = 'Alabama'",
    ),
    ("SELECT AVG(age) FROM head", "SELECT AVG( T1.age ) FROM head AS T1;"),
    ("SELECT COUNT(DISTINCT born_state) FROM head", "SELECT COUNT(DISTINCT(h.born_state)) FROM head h"),
]

WRONG = [
    ("SELECT COUNT(*) FROM head WHERE age > 56", "SELECT COUNT(*) FROM head WHERE age > 10"),
    ("SELECT COUNT(*) FROM head WHERE age > 56", "SELECT name FROM head WHERE age > 56"),
    ("SELECT name FROM head WHERE born_state = 'Alabama'", "SELECT name FROM head"),
    ("SELECT AVG(age) FROM head", "SELECT SUM(age) FROM head"),
]


def test_parse_schema_finds_tables_and_columns():
    tables = parse_schema(SCHEMA)
    assert set(tables) == {"head", "department"}
    assert set(tables["head"]) == {"head_id", "name", "age", "born_state"}


def test_gold_against_itself_scores_100_percent():
    for gold in GOLD_CASES:
        result = execution_match(SCHEMA, gold, gold)
        assert result.gold_ok, (gold, result.gold_error)
        assert result.is_valid
        assert result.execution_match, gold


def test_equivalent_rewrites_match():
    for gold, rewrite in EQUIVALENT:
        result = execution_match(SCHEMA, gold, rewrite)
        assert result.is_valid, (rewrite, result.pred_error)
        assert result.execution_match, (gold, rewrite)


def test_wrong_queries_do_not_match():
    for gold, wrong in WRONG:
        result = execution_match(SCHEMA, gold, wrong)
        assert result.is_valid, (wrong, result.pred_error)
        assert not result.execution_match, (gold, wrong)


def test_malformed_sql_is_invalid_not_a_crash():
    for bad in ["SELEC * FROM head", "SELECT * FROM", "SELECT * FROM no_such_table"]:
        result = execution_match(SCHEMA, GOLD_CASES[0], bad)
        assert result.is_valid is False
        assert result.execution_match is False
        assert result.pred_error or result.pred_reason


def test_empty_prediction_is_invalid():
    result = execution_match(SCHEMA, GOLD_CASES[0], "")
    assert result.is_valid is False


def test_write_statements_are_rejected_as_not_read_only():
    conn, _ = build_database(SCHEMA, GOLD_CASES[0])
    result = execute_sql(conn, "DELETE FROM head")
    assert result.ok is False
    assert "not_read_only" in (result.reason or "")
    conn.close()


def test_order_by_detection_and_ordering_sensitivity():
    assert has_order_by("SELECT name FROM head ORDER BY age")
    assert not has_order_by("SELECT name FROM head")
    gold = "SELECT name FROM head ORDER BY age DESC LIMIT 3"
    shuffled = "SELECT name FROM head LIMIT 3"
    assert execution_match(SCHEMA, gold, shuffled).order_sensitive is True


def test_runaway_query_times_out():
    conn, _ = build_database(SCHEMA, GOLD_CASES[0])
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT count(*) FROM c"
    )
    result = execute_sql(conn, sql, timeout_s=0.05)
    conn.close()
    assert result.ok is False
    assert result.reason in ("timeout", "sqlite_error")


def test_synthetic_database_is_deterministic():
    first, info_a = build_database(SCHEMA, GOLD_CASES[1], seed=7)
    second, info_b = build_database(SCHEMA, GOLD_CASES[1], seed=7)
    assert info_a["tables"] == info_b["tables"]
    first.close()
    second.close()


def test_gold_literals_are_present_in_synthetic_rows():
    gold = "SELECT name FROM head WHERE born_state = 'Alabama'"
    conn, info = build_database(SCHEMA, gold)
    rows = info["tables"]["head"]["rows"]
    states = {row[3] for row in rows}
    assert "Alabama" in states
    conn.close()
