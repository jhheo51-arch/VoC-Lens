"""감정 급변 구간 탐지.

지수 시계열을 지수이동평균(EMA)으로 평활화한 뒤, window칸 전 대비 변화량이 min_delta 이상인
연속 구간을 하나의 이벤트로 묶는다. 같은 방향으로 이어지는 후보는 합치고 가장 큰 변화량을 남긴다.
4주차 이후 데이터가 쌓이면 ruptures의 변화점 탐지와 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ema(values, alpha: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for i, v in enumerate(values):
        out[i] = v if i == 0 else alpha * v + (1 - alpha) * out[i - 1]
    return out


@dataclass
class Change:
    start: int      # 변화가 시작된 위치 (window칸 전)
    end: int        # 변화가 가장 컸던 위치
    delta: float


def detect_changes(values, window: int = 2, min_delta: float = 0.35, alpha: float = 0.5) -> list[Change]:
    s = ema(values, alpha)
    if len(s) <= window:
        return []
    deltas = np.zeros(len(s))
    deltas[window:] = s[window:] - s[:-window]

    changes: list[Change] = []
    run: list[int] = []
    for i in range(window, len(s)):
        d = deltas[i]
        if abs(d) >= min_delta and (not run or np.sign(d) == np.sign(deltas[run[-1]])):
            run.append(i)
            continue
        if run:
            changes.append(_close(run, deltas, window))
            run = []
        if abs(d) >= min_delta:
            run = [i]
    if run:
        changes.append(_close(run, deltas, window))
    return changes


def _close(run: list[int], deltas: np.ndarray, window: int) -> Change:
    peak = max(run, key=lambda i: abs(deltas[i]))
    return Change(start=max(0, run[0] - window), end=peak, delta=round(float(deltas[peak]), 4))
