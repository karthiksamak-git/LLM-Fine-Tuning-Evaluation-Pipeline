"""Statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mlfte.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    compare_metric,
    mcnemar_test,
    mean_std,
    relative_improvement,
)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    first = bootstrap_ci(values, n_resamples=500, seed=3)
    second = bootstrap_ci(values, n_resamples=500, seed=3)
    assert first == second
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["mean"] == pytest.approx(np.mean(values))


def test_bootstrap_constant_values_have_zero_width():
    ci = bootstrap_ci([1.0] * 20, n_resamples=200, seed=1)
    assert ci["lo"] == ci["hi"] == 1.0


def test_bootstrap_handles_empty_and_single():
    assert np.isnan(bootstrap_ci([])["mean"])
    assert bootstrap_ci([1.0])["n"] == 1


def test_mcnemar_exact_small_counts():
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2
    result = mcnemar_test(a, b)
    assert result["b"] == 10
    assert result["c"] == 2
    assert result["method"].startswith("exact")
    assert result["p_value"] < 0.05


def test_mcnemar_chi_square_for_large_counts():
    a = [True] * 40 + [False] * 40
    b = [False] * 40 + [True] * 40
    result = mcnemar_test(a, b)
    assert result["method"].startswith("chi-square")
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_degenerate_when_identical():
    a = [True, False, True]
    result = mcnemar_test(a, list(a))
    assert result["p_value"] == 1.0
    assert result["n_discordant"] == 0


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])


def test_effect_sizes_and_relative_improvement():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.5, 0.8) > 0
    assert relative_improvement(0.2, 0.3) == pytest.approx(0.5)
    assert relative_improvement(0.0, 0.3) is None


def test_compare_metric_structure():
    old = [True] * 10 + [False] * 10
    new = [True] * 14 + [False] * 6
    result = compare_metric(old, new, n_resamples=300, seed=0)
    assert result["absolute_difference"] == pytest.approx(0.2)
    assert "mcnemar" in result
    assert result["baseline"]["n"] == 20


def test_mean_std():
    assert mean_std([1.0, 3.0])["mean"] == 2.0
    assert mean_std([1.0, 3.0])["std"] == pytest.approx(2**0.5)
    assert mean_std([])["n"] == 0
