"""세 채널(리뷰·상담·영상)이 공유하는 분석 단위 스키마.

모든 분석 단계는 list[Unit]을 받아 자기 담당 필드만 채워서 돌려준다.
자세한 정의는 docs/analysis_design.md 0.2절.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

EMOTIONS = ("기쁨", "슬픔", "분노", "불안", "놀람", "혐오", "중립")

INDEX_NAMES = ("stress", "happiness", "negative_affect", "frustration", "recovery", "engagement")
INDEX_LABELS = {
    "stress": "스트레스",
    "happiness": "행복",
    "negative_affect": "부정 정서",
    "frustration": "좌절",
    "recovery": "감정 회복",
    "engagement": "몰입",
}


class SourceType(str, Enum):
    REVIEW = "review"
    CALL = "call"
    VIDEO = "video"


class Unit(BaseModel):
    """최소 분석 단위. 리뷰는 절, 상담·영상은 발화."""

    unit_id: str
    session_id: str
    source_type: SourceType
    order: int
    text: str
    clean_text: str = ""
    speaker_role: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None

    valence: float | None = None
    arousal: float | None = None
    emotion_probs: dict[str, float] = Field(default_factory=dict)

    aspect: str | None = None
    aspect_sentiment: float | None = None
    complaint_categories: list[str] = Field(default_factory=list)
    journey_stage: str | None = None
    cx_signals: dict[str, bool] = Field(default_factory=dict)

    # 단위 점수 (0~1). 세션 지수는 이것을 모아서 만든다.
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)


class IndexSummary(BaseModel):
    """세션 단위 지수 요약. 값은 0~100."""

    name: str
    raw: float | None
    mean: float | None = None
    peak: float | None = None
    above_ratio: float | None = None
    end_value: float | None = None
    percentile: float | None = None


class Event(BaseModel):
    """감정 급변 구간. delta > 0이면 상승(예: 회복), < 0이면 하락."""

    event_id: str
    index_name: str
    start_sec: float | None
    end_sec: float | None
    delta: float
    start_unit_id: str
    end_unit_id: str
    # 급변 직전의 상대방 발화 (상담: 고객 감정 변화 직전의 상담원 발화)
    trigger_unit_id: str | None = None


class Session(BaseModel):
    """리뷰 1건 / 통화 1건 / 영상 1건."""

    session_id: str
    source_id: str
    source_type: SourceType
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    units: list[Unit] = Field(default_factory=list)
    indices: dict[str, IndexSummary] = Field(default_factory=dict)
    peak_end: float | None = None
    events: list[Event] = Field(default_factory=list)
