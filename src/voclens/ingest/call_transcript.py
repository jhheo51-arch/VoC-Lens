"""상담 전사본(JSON) 읽기.

형식 (docs/call_transcript_format.md)
{
  "call_id": "call_001",
  "meta": {"brand": "...", "domain": "배달", "created_at": "2026-10-01"},
  "audio": "선택: 같은 폴더 기준 WAV 경로",
  "turns": [
    {"speaker": "A", "start": 0.0, "end": 3.2, "text": "...", "role": "선택: agent|customer", "stt_conf": 0.9}
  ]
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Turn(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    role: str | None = None
    stt_conf: float | None = None

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end({v})가 start({start})보다 작습니다")
        return v


class CallTranscript(BaseModel):
    call_id: str
    turns: list[Turn]
    meta: dict = Field(default_factory=dict)
    audio: str | None = None
    source_path: str | None = None

    @property
    def duration(self) -> float:
        return max((t.end for t in self.turns), default=0.0)

    def audio_path(self) -> Path | None:
        if not self.audio:
            return None
        path = Path(self.audio)
        if not path.is_absolute() and self.source_path:
            path = Path(self.source_path).parent / path
        return path


_ROLE_WORDS = {
    "상담원": "agent", "상담사": "agent", "직원": "agent", "agent": "agent", "a": None,
    "고객": "customer", "손님": "customer", "customer": "customer", "b": None,
}
_LINE = re.compile(r"^\s*(?:\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?(?P<spk>[^:：]{1,20})\s*[:：]\s*(?P<text>.+)$")
# 시각이 없을 때 발화 길이를 추정하는 값 (한국어 상담 말 속도 대략 초당 5~6음절)
SEC_PER_CHAR = 0.18
GAP_SEC = 0.6


def _ts_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    return float(parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0))


def parse_text_transcript(text: str, call_id: str = "pasted_call", meta: dict | None = None) -> CallTranscript:
    """'화자: 발화' 줄로 된 전사본을 읽는다. 줄 앞에 [mm:ss] 시각이 있으면 쓰고, 없으면 글자 수로 추정한다.

    화자 이름이 상담원·고객(상담사, 손님 등)이면 역할을 그대로 쓰고, 그 밖의 이름(A, B 등)은 역할 판정 규칙에 맡긴다.
    화자 표시가 없는 줄은 앞 발화에 이어 붙인다.
    """
    rows: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            if rows:
                rows[-1]["text"] += " " + line
            continue
        rows.append({"speaker": m.group("spk").strip(), "text": m.group("text").strip(), "ts": m.group("ts")})
    if not rows:
        raise ValueError("'화자: 발화' 형식의 줄이 없습니다. 예: 상담원: 무엇을 도와드릴까요")

    turns = []
    clock = 0.0
    for i, r in enumerate(rows):
        start = _ts_seconds(r["ts"]) if r["ts"] else clock
        duration = max(1.0, len(r["text"].replace(" ", "")) * SEC_PER_CHAR)
        next_ts = rows[i + 1]["ts"] if i + 1 < len(rows) else None
        end = min(start + duration, _ts_seconds(next_ts)) if next_ts and _ts_seconds(next_ts) > start else start + duration
        role = _ROLE_WORDS.get(r["speaker"].lower())
        turns.append(Turn(speaker=r["speaker"], start=round(start, 2), end=round(end, 2), text=r["text"], role=role))
        clock = end + GAP_SEC
    # 모든 화자에 역할이 있을 때만 그대로 쓰고, 일부만 있으면 판정 규칙이 모두 다시 정한다
    if not all(t.role for t in turns):
        for t in turns:
            t.role = None
    return CallTranscript(call_id=call_id, turns=turns, meta=meta or {})


def load_transcript(path: str | Path) -> CallTranscript:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("call_id", path.stem)
    data["source_path"] = str(path)
    call = CallTranscript.model_validate(data)
    call.turns = sorted((t for t in call.turns if t.text.strip()), key=lambda t: t.start)
    if not call.turns:
        raise ValueError(f"{path}: 내용이 있는 발화가 없습니다")
    return call
