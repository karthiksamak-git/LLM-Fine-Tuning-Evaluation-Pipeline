"""Metrics: normalized exact match, SQL validity, execution match, bucket breakdown."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from mlfte.config import Config
from mlfte.evaluation.sql_exec import execution_match, limitations
from mlfte.evaluation.stats import bootstrap_ci
from mlfte.tasks.base import Example, Task

_WS_RE = re.compile(r"\s+")
_COMMENT_RE = re.compile(r"--[^\n]*")


def normalize_sql(sql: str) -> str:
    """Canonical form for exact match: lowercase keywords, collapsed whitespace, no
    trailing semicolon, resolved quoting when sqlglot can parse it."""
    text = _COMMENT_RE.sub(" ", sql or "")
    text = _WS_RE.sub(" ", text).strip().rstrip(";").strip()
    if not text:
        return ""
    try:
        import sqlglot

        tree = sqlglot.parse_one(text, dialect="sqlite")
        return _WS_RE.sub(" ", tree.sql(dialect="sqlite")).strip().lower()
    except Exception:
        return text.lower()


def exact_match(pred: str, gold: str) -> bool:
    return bool(pred) and normalize_sql(pred) == normalize_sql(gold)


def score_item(
    task: Task,
    example: Example,
    prediction: str,
    cfg: Config,
    seed: int = 0,
    check_execution: bool = True,
) -> dict[str, Any]:
    """Score one prediction: exact match, validity, execution match, bucket."""
    parsed = task.parse_output(prediction)
    em = exact_match(parsed, example.answer)
    item: dict[str, Any] = {
        "id": example.id,
        "bucket": task.bucket(example),
        "prediction": parsed,
        "gold": example.answer,
        "exact_match": em,
        "is_valid": None,
        "execution_match": None,
        "execution_reason": None,
        "gold_execution_ok": None,
        "pred_error": None,
    }
    if check_execution:
        comparison = execution_match(
            example.context,
            example.answer,
            parsed,
            seed=seed,
            timeout_s=cfg.eval.exec_timeout_s,
        )
        item.update(
            is_valid=comparison.is_valid,
            execution_match=comparison.execution_match,
            execution_reason=comparison.reason,
            gold_execution_ok=comparison.gold_ok,
            pred_error=comparison.pred_error,
        )
    return item


def _metric_summary(values: list[bool], n_resamples: int, seed: int) -> dict[str, Any]:
    floats = [1.0 if v else 0.0 for v in values]
    ci = bootstrap_ci(floats, n_resamples=n_resamples, seed=seed)
    return {"value": ci["mean"], "ci_low": ci["lo"], "ci_high": ci["hi"], "n": ci["n"]}


def aggregate_items(
    items: list[dict[str, Any]], cfg: Config, seed: int = 0
) -> dict[str, Any]:
    """Aggregate per-item scores into overall + per-bucket metrics with CIs."""
    valid_items = [it for it in items if it["execution_match"] is not None]
    def collect(key: str, rows: list[dict[str, Any]]) -> list[bool]:
        return [bool(r[key]) for r in rows if r[key] is not None]

    metrics = {}
    for key in ("exact_match", "is_valid", "execution_match"):
        metrics[key] = _metric_summary(collect(key, valid_items), cfg.eval.bootstrap_resamples, seed)

    buckets: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in valid_items:
        grouped[item["bucket"]].append(item)
    for bucket, rows in sorted(grouped.items()):
        buckets[bucket] = {
            "n": len(rows),
            "exact_match": _metric_summary(
                collect("exact_match", rows), cfg.eval.bootstrap_resamples, seed
            ),
            "is_valid": _metric_summary(
                collect("is_valid", rows), cfg.eval.bootstrap_resamples, seed
            ),
            "execution_match": _metric_summary(
                collect("execution_match", rows), cfg.eval.bootstrap_resamples, seed
            ),
        }

    gold_failures = sum(1 for it in valid_items if it.get("gold_execution_ok") is False)
    return {
        "n": len(valid_items),
        "metrics": metrics,
        "by_bucket": buckets,
        "gold_execution_failures": gold_failures,
        "execution_match_note": (
            "Execution match uses synthetic seeded SQLite data; see sql_exec limitations."
        ),
        "limitations": limitations(),
    }
