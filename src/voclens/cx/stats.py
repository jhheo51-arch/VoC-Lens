"""집계 통계: 소표본 보정, 신뢰구간, 그룹 간 차이 검정, 별점 불일치."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def shrink_mean(mean: float, n: int, global_mean: float, k: float = 10) -> float:
    """베이지안 축소: 표본이 적을수록 전체 평균 쪽으로 당긴다. k는 사전분포의 가상 표본 수."""
    if n <= 0:
        return global_mean
    return (n * mean + k * global_mean) / (n + k)


def bootstrap_ci(values, iterations: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(iterations, len(arr)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_label: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _cramers_v_label(v: float, dof: int) -> str:
    # Cohen(1988) 기준, 자유도에 따라 기준값이 달라진다
    small, medium, large = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35)}.get(min(dof, 2), (0.06, 0.17, 0.29))
    if v >= large:
        return "큼"
    if v >= medium:
        return "중간"
    if v >= small:
        return "작음"
    return "거의 없음"


def compare_proportions(table) -> TestResult:
    """그룹 × 범주 빈도표의 카이제곱 독립성 검정과 Cramér's V."""
    observed = np.asarray(table, dtype=float)
    observed = observed[:, observed.sum(axis=0) > 0]
    observed = observed[observed.sum(axis=1) > 0]
    chi2, p, dof, _ = stats.chi2_contingency(observed)
    n = observed.sum()
    k = min(observed.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else 0.0
    return TestResult("카이제곱 검정", float(chi2), float(p), v, _cramers_v_label(v, min(dof, k)))


def compare_distributions(a, b) -> TestResult:
    """두 그룹 지수 분포의 Mann-Whitney U 검정과 순위 이연 상관(rank-biserial)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = float(1 - 2 * u / (len(a) * len(b)))
    size = abs(r)
    label = "큼" if size >= 0.5 else "중간" if size >= 0.3 else "작음" if size >= 0.1 else "거의 없음"
    return TestResult("Mann-Whitney U", float(u), float(p), r, label)


def rating_mismatch(ratings, scores, z_threshold: float = 2.0) -> np.ndarray:
    """감정 점수로 별점을 예측하는 1차 회귀의 표준화 잔차. |z|가 큰 리뷰가 별점–감정 불일치 후보.

    반환값: 각 리뷰의 표준화 잔차 (양수 = 감정에 비해 별점이 높음).
    """
    x = np.asarray(scores, dtype=float)
    y = np.asarray(ratings, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]):
        return np.zeros(len(x))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    sd = residual.std(ddof=1)
    return residual / sd if sd > 0 else np.zeros(len(x))
