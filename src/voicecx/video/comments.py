"""댓글 분석 보조: 타임스탬프 추출과 구간 연결, 크리에이터 vs 시청자 감정 비교."""

from __future__ import annotations

import re

import numpy as np

# "3:20", "03:20", "1:02:03". 앞뒤에 숫자나 콜론이 붙어 있으면 제외한다. 초가 60 이상이면 시각이 아니다.
_TIMESTAMP = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def extract_timestamps(text: str, max_sec: float | None = None) -> list[float]:
    out = []
    for h, m, s in _TIMESTAMP.findall(text):
        minutes, seconds = int(m), int(s)
        if seconds >= 60 or (not h and minutes >= 60):
            continue
        value = int(h or 0) * 3600 + minutes * 60 + seconds
        if max_sec is None or value <= max_sec:
            out.append(float(value))
    return out


def find_segment(timestamp: float, spans: list[tuple[float, float]], tolerance: float = 5.0) -> int | None:
    """시각이 들어가는 구간 인덱스. 정확히 안 맞으면 tolerance 안에서 가장 가까운 구간."""
    for i, (start, end) in enumerate(spans):
        if start <= timestamp <= end:
            return i
    best, best_dist = None, tolerance
    for i, (start, end) in enumerate(spans):
        dist = min(abs(timestamp - start), abs(timestamp - end))
        if dist <= best_dist:
            best, best_dist = i, dist
    return best


def compare_creator_audience(creator_valence: list[float], audience_valence: list[float]) -> dict:
    """크리에이터와 시청자의 평균 Valence 차이. 양수면 시청자가 더 긍정적."""
    c = float(np.mean(creator_valence)) if creator_valence else None
    a = float(np.mean(audience_valence)) if audience_valence else None
    gap = None if c is None or a is None else round(a - c, 4)
    return {
        "creator_valence": None if c is None else round(c, 4),
        "audience_valence": None if a is None else round(a, 4),
        "audience_minus_creator": gap,
        "n_creator": len(creator_valence),
        "n_audience": len(audience_valence),
    }
