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
