"""Statistics: bootstrap confidence intervals, McNemar's paired test, effect sizes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of ``values`` (0/1 accuracy or continuous)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "lo": float(arr[0]), "hi": float(arr[0]), "n": 1}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(arr.size),
        "n_resamples": int(n_resamples),
        "alpha": alpha,
    }


def mcnemar_test(
    correct_a: list[bool] | np.ndarray, correct_b: list[bool] | np.ndarray
) -> dict[str, Any]:
    """Paired McNemar test between two models evaluated on the same items.

    ``b`` = items where A is right and B is wrong; ``c`` = the reverse. Uses the exact
    two-sided binomial test when ``b + c < 25`` and the continuity-corrected chi-square
    otherwise.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    if a.shape != b_arr.shape:
        raise ValueError("paired arrays must have the same length")
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    n = b + c
    if n == 0:
        return {
            "b": 0,
            "c": 0,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "degenerate (models agree on every item)",
        }
    if n < 25:
        p = float(scipy_stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        return {
            "b": b,
            "c": c,
            "n_discordant": n,
            "statistic": float(min(b, c)),
            "p_value": p,
            "method": "exact binomial (two-sided)",
        }
    statistic = (abs(b - c) - 1) ** 2 / n
    p = float(scipy_stats.chi2.sf(statistic, df=1))
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "statistic": float(statistic),
        "p_value": p,
        "method": "chi-square with continuity correction (df=1)",
    }


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size: positive when ``p2`` (after) is larger than ``p1`` (before)."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return float(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)))


def relative_improvement(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return float((after - before) / before)


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def compare_metric(
    correct_base: list[bool],
    correct_new: list[bool],
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap CI per model plus paired McNemar test and effect size for one metric."""
    base = bootstrap_ci(correct_base, n_resamples, seed=seed)
    new = bootstrap_ci(correct_new, n_resamples, seed=seed + 1)
    return {
        "baseline": base,
        "model": new,
        "absolute_difference": new["mean"] - base["mean"],
        "relative_improvement": relative_improvement(base["mean"], new["mean"]),
        "mcnemar": mcnemar_test(correct_base, correct_new),
        "cohens_h": cohens_h(base["mean"], new["mean"]),
    }
