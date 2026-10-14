"""6개 지수 계산.

단위 점수(0~1)는 Valence·Arousal과 CX 신호로 만들고, 세션 지수(0~100)는 단위 점수를 모아 만든다.
지수 정의는 PRD 7장, 상수의 의미는 아래 주석 참고.

  스트레스     부정 × 고각성     max(0,-V) × A / 0.6
  행복         긍정              max(0,V) / 0.8
  부정 정서    부정 × 저각성     max(0,-V) × (1-A) / 0.5
  좌절         반복 문의·대기·이탈 신호 + 부정 강도
  감정 회복    후반부 평균 V - 전반부 평균 V (세션 단위만)
  몰입         글 길이·언급한 속성 수·재주문/추천 의사 (세션 단위만)

나누는 상수(0.6, 0.8, 0.5)는 Circumplex 좌표에서 해당 감정만 강하게 나왔을 때 1 근처가 되도록 잡은 값이다.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import percentileofscore

from voicecx.schema import INDEX_NAMES, IndexSummary, Session, Unit

UNIT_LEVEL = ("stress", "happiness", "negative_affect", "frustration")


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def unit_scores(unit: Unit) -> dict[str, float]:
    v = unit.valence or 0.0
    a = unit.arousal if unit.arousal is not None else 0.3
    s = unit.cx_signals
    negative = max(0.0, -v)

    frustration = (
        0.40 * s.get("repeat_contact", False)
        + 0.25 * s.get("waiting", False)
        + 0.15 * (s.get("churn_intent", False) or s.get("not_recommend", False))
        + 0.40 * negative
    )
    return {
        "stress": round(_clip(negative * a / 0.6), 4),
        "happiness": round(_clip(max(0.0, v) / 0.8), 4),
        "negative_affect": round(_clip(negative * (1 - a) / 0.5), 4),
        "frustration": round(_clip(frustration), 4),
    }


def _summarize(name: str, values: list[float], threshold: float) -> IndexSummary:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    return IndexSummary(
        name=name,
        raw=round(mean * 100, 2),
        mean=round(mean * 100, 2),
        peak=round(float(arr.max()) * 100, 2),
        above_ratio=round(float((arr >= threshold).mean()), 4),
        end_value=round(float(arr[-1]) * 100, 2),
    )


def recovery_index(units: list[Unit], min_negative: float = -0.2) -> float | None:
    """전반부에 부정 감정이 있었을 때만 계산. 50 = 변화 없음, 100 = 완전 회복, 0 = 악화."""
    valences = [u.valence or 0.0 for u in units]
    if len(valences) < 2:
        return None
    half = len(valences) // 2
    first, second = valences[:half], valences[half:]
    if min(first) > min_negative:
        return None
    delta = float(np.mean(second) - np.mean(first))
    return round(50 + 50 * max(-1.0, min(1.0, delta)), 2)


def engagement_index(units: list[Unit]) -> float:
    total_chars = sum(len(u.text) for u in units)
    aspects = {u.aspect for u in units if u.aspect}
    intent = any(
        u.cx_signals.get("repurchase_intent") or u.cx_signals.get("recommend_intent") for u in units
    )
    score = 0.35 * min(total_chars / 150, 1) + 0.35 * min(len(aspects) / 3, 1) + 0.30 * intent
    return round(_clip(score) * 100, 2)


def score_session(
    session: Session, threshold: float = 0.5, min_negative: float = -0.2, units: list[Unit] | None = None
) -> Session:
    """세션 지수를 계산한다. units를 주면 그 발화만으로 지수를 만든다 (상담: 고객 발화만)."""
    for unit in session.units:
        unit.scores = unit_scores(unit)

    target = session.units if units is None else units
    if not target:
        session.indices = {name: IndexSummary(name=name, raw=None) for name in INDEX_NAMES}
        return session

    indices = {name: _summarize(name, [u.scores[name] for u in target], threshold) for name in UNIT_LEVEL}
    indices["recovery"] = IndexSummary(name="recovery", raw=recovery_index(target, min_negative))
    indices["engagement"] = IndexSummary(name="engagement", raw=engagement_index(target))
    session.indices = indices

    # Peak-End: 가장 나빴던 순간과 마지막 순간의 부정 경험 평균 (높을수록 나쁜 기억)
    negative = [max(u.scores["stress"], u.scores["negative_affect"], u.scores["frustration"]) for u in target]
    session.peak_end = round((max(negative) + negative[-1]) / 2 * 100, 2)
    return session


def assign_percentiles(sessions: list[Session]) -> None:
    """같은 채널 안에서의 백분위. 채널마다 감정 분포가 달라서 원점수 대신 비교에 쓴다."""
    by_type: dict[str, list[Session]] = {}
    for s in sessions:
        by_type.setdefault(s.source_type.value, []).append(s)
    for group in by_type.values():
        for name in INDEX_NAMES:
            values = [s.indices[name].raw for s in group if name in s.indices and s.indices[name].raw is not None]
            if not values:
                continue
            for s in group:
                summary = s.indices.get(name)
                if summary and summary.raw is not None:
                    summary.percentile = round(float(percentileofscore(values, summary.raw, kind="mean")), 1)
