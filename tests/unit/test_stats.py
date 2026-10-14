import numpy as np
import pytest

from voicecx.cx.stats import bootstrap_ci, compare_distributions, compare_proportions, rating_mismatch, shrink_mean


def test_shrink_mean_pulls_small_samples():
    assert shrink_mean(100, 1, 50, k=10) == pytest.approx(54.545, abs=1e-3)
    assert shrink_mean(100, 1000, 50, k=10) == pytest.approx(99.505, abs=1e-3)
    assert shrink_mean(100, 0, 50) == 50


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = np.random.default_rng(0).normal(10, 2, 200)
    low, high = bootstrap_ci(values, seed=1)
    assert low < values.mean() < high
    assert bootstrap_ci(values, seed=1) == (low, high)


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci([5.0]) == (5.0, 5.0)
    assert all(np.isnan(bootstrap_ci([])))


def test_compare_proportions_detects_difference():
    same = compare_proportions([[50, 50], [50, 50]])
    diff = compare_proportions([[90, 10], [10, 90]])
    assert same.p_value > 0.5 and same.effect_size == pytest.approx(0.0)
    assert diff.significant and diff.effect_label == "큼"


def test_compare_proportions_drops_empty_columns():
    result = compare_proportions([[10, 0, 5], [3, 0, 12]])
    assert 0 <= result.effect_size <= 1


def test_compare_distributions():
    a = np.arange(0, 50)
    b = np.arange(30, 80)
    result = compare_distributions(a, b)
    assert result.significant
    assert result.effect_size > 0  # a가 작을수록 양수
    assert compare_distributions(a, a).p_value > 0.9


def test_rating_mismatch_flags_outlier():
    scores = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.1]
    ratings = [1, 2, 3, 4, 5, 5, 5]  # 마지막: 감정은 낮은데 별점 5점
    z = rating_mismatch(ratings, scores)
    assert int(np.argmax(z)) == 6
    assert z[6] > 1.5


def test_rating_mismatch_constant_scores():
    assert (rating_mismatch([1, 5, 3], [0.5, 0.5, 0.5]) == 0).all()
