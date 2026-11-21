"""Failure taxonomy and curated side-by-side examples (base vs few-shot vs fine-tuned)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import sqlglot
from sqlglot import exp

from mlfte.tasks.base import Example

TAXONOMY = (
    "extraction_failure",
    "invalid_sql",
    "wrong_table_or_column",
    "wrong_aggregation",
    "wrong_filter_value",
    "missing_or_extra_clause",
    "result_mismatch_other",
)


def _parse(sql: str):
    try:
        return sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None


def _aggregations(tree) -> set[str]:
    if tree is None:
        return set()
    return {type(node).__name__ for node in tree.find_all(exp.AggFunc)}


def _clauses(tree) -> set[str]:
    if tree is None:
        return set()
    found = set()
    for cls in (exp.Join, exp.Where, exp.Group, exp.Order, exp.Limit, exp.Having, exp.Distinct):
        if any(True for _ in tree.find_all(cls)):
            found.add(cls.__name__)
    return found


def _literals(tree) -> set[str]:
    if tree is None:
        return set()
    return {str(node.this).lower() for node in tree.find_all(exp.Literal)}


def classify_failure(item: dict[str, Any]) -> str | None:
    """Heuristic label for a *failed* prediction (``None`` when it passed)."""
    if item.get("execution_match") or item.get("exact_match"):
        return None
    pred = item.get("prediction") or ""
    if not pred.strip():
        return "extraction_failure"
    if item.get("is_valid") is False:
        error = (item.get("pred_error") or "").lower()
        if "no such table" in error or "no such column" in error:
            return "wrong_table_or_column"
        return "invalid_sql"
    gold_tree = _parse(item.get("gold", ""))
    pred_tree = _parse(pred)
    if pred_tree is None:
        return "invalid_sql"
    if _aggregations(pred_tree) != _aggregations(gold_tree):
        return "wrong_aggregation"
    if _literals(pred_tree) != _literals(gold_tree):
        return "wrong_filter_value"
    if _clauses(pred_tree) != _clauses(gold_tree):
        return "missing_or_extra_clause"
    return "result_mismatch_other"


def taxonomy_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        label = classify_failure(item)
        counts[label or "correct"] += 1
    return dict(counts)


def _outcome(run_items: dict[str, dict[str, Any]], example_id: str) -> dict[str, Any]:
    row = {}
    for label, items in run_items.items():
        item = items.get(example_id)
        row[label] = {
            "prediction": item.get("prediction") if item else None,
            "execution_match": bool(item.get("execution_match")) if item else None,
            "exact_match": bool(item.get("exact_match")) if item else None,
            "is_valid": item.get("is_valid") if item else None,
            "failure_type": classify_failure(item) if item else None,
        }
    return row


def analyze(
    examples: list[Example],
    runs: dict[str, list[dict[str, Any]]],
    primary: str,
    baseline: str,
    max_examples: int = 200,
) -> dict[str, Any]:
    """Taxonomy per run plus fixed/regressed/still-wrong selections.

    ``runs`` maps a label (``base_zero_shot``, ``base_few_shot``, ``finetuned``) to scored
    items. ``primary`` is compared against ``baseline`` to bucket outcomes.
    """
    by_id = {label: {item["id"]: item for item in items} for label, items in runs.items()}
    taxonomy = {label: taxonomy_counts(items) for label, items in runs.items()}

    example_by_id = {ex.id: ex for ex in examples}
    primary_items = by_id.get(primary, {})
    baseline_items = by_id.get(baseline, {})

    buckets: dict[str, list[str]] = defaultdict(list)
    for example_id, item in primary_items.items():
        base = baseline_items.get(example_id)
        if base is None:
            continue
        base_ok = bool(base.get("execution_match"))
        new_ok = bool(item.get("execution_match"))
        if not base_ok and new_ok:
            buckets["fixed"].append(example_id)
        elif base_ok and not new_ok:
            buckets["regressed"].append(example_id)
        elif not new_ok:
            buckets["still_wrong"].append(example_id)
        else:
            buckets["still_correct"].append(example_id)

    selected: list[dict[str, Any]] = []
    for category in ("fixed", "regressed", "still_wrong"):
        for example_id in buckets[category][: max_examples // 3 or 1]:
            ex = example_by_id.get(example_id)
            if ex is None:
                continue
            selected.append(
                {
                    "id": example_id,
                    "category": category,
                    "question": ex.question,
                    "schema": ex.context,
                    "gold": ex.answer,
                    "bucket": {item["bucket"] for item in primary_items.values()
                               if item["id"] == example_id} or {"all"},
                    "models": _outcome(by_id, example_id),
                }
            )

    return {
        "taxonomy": taxonomy,
        "taxonomy_labels": list(TAXONOMY) + ["correct"],
        "outcomes": {k: len(v) for k, v in buckets.items()},
        "primary": primary,
        "baseline": baseline,
        "examples": selected[:max_examples],
        "example_pool": {k: v[:50] for k, v in buckets.items()},
        "method": (
            "Heuristic taxonomy from SQL structure: empty output -> extraction_failure; "
            "execution error -> invalid_sql / wrong_table_or_column; differing aggregate "
            "functions, literals or clause sets explain the rest; otherwise "
            "result_mismatch_other."
        ),
    }
