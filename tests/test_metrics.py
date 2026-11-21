"""Metric helpers and SQL extraction."""

from __future__ import annotations

from mlfte.evaluation.error_analysis import classify_failure, taxonomy_counts
from mlfte.evaluation.metrics import exact_match, normalize_sql
from mlfte.tasks import get_task
from mlfte.tasks.base import Example


def test_normalize_sql_ignores_case_whitespace_and_semicolons():
    assert normalize_sql("SELECT  a  FROM t;") == normalize_sql("select a from t")


def test_exact_match_positive_and_negative():
    assert exact_match("SELECT a FROM t", "select a from t;")
    assert not exact_match("SELECT a FROM t", "SELECT b FROM t")
    assert not exact_match("", "SELECT a FROM t")


def test_parse_output_extracts_sql_from_fences_and_prose():
    task = get_task()
    text = "```sql\nSELECT COUNT(*) FROM head;\n```\nThis counts the rows."
    assert task.parse_output(text) == "SELECT COUNT(*) FROM head;"
    assert task.parse_output("Here you go: SELECT 1") == "SELECT 1"
    assert task.parse_output("") == ""
    assert task.parse_output("<|im_end|>") == ""


def test_parse_output_stops_at_statement_terminator():
    task = get_task()
    assert task.parse_output("SELECT 1; DROP TABLE head;") == "SELECT 1;"


def test_classify_failure_categories():
    assert classify_failure({"prediction": "", "execution_match": False}) == "extraction_failure"
    assert (
        classify_failure(
            {"prediction": "SELECT 1", "is_valid": False, "pred_error": "no such table: x"}
        )
        == "wrong_table_or_column"
    )
    assert (
        classify_failure({"prediction": "SELECT ( FROM", "is_valid": False, "pred_error": "syntax"})
        == "invalid_sql"
    )
    assert (
        classify_failure(
            {
                "prediction": "SELECT SUM(a) FROM t",
                "gold": "SELECT COUNT(a) FROM t",
                "is_valid": True,
                "execution_match": False,
            }
        )
        == "wrong_aggregation"
    )
    assert (
        classify_failure(
            {
                "prediction": "SELECT a FROM t WHERE a > 5",
                "gold": "SELECT a FROM t WHERE a > 3",
                "is_valid": True,
                "execution_match": False,
            }
        )
        == "wrong_filter_value"
    )
    assert classify_failure({"execution_match": True}) is None


def test_taxonomy_counts_shape():
    items = [
        {"execution_match": True, "exact_match": True},
        {"prediction": "", "execution_match": False},
    ]
    counts = taxonomy_counts(items)
    assert counts["correct"] == 1
    assert counts["extraction_failure"] == 1


def test_complexity_features_on_example():
    task = get_task()
    example = Example(
        "x",
        "q",
        "CREATE TABLE t (a INT); CREATE TABLE u (b INT);",
        "SELECT COUNT(t.a) FROM t JOIN u ON t.a = u.b WHERE t.a > 1 GROUP BY t.a",
    )
    features = task.complexity(example)
    assert features["joins"] >= 1
    assert features["aggregations"] >= 1
    assert features["where"] == 1
    assert features["group_by"] == 1
