"""음성 파일 → 화자가 붙은 발화 목록 (전사본이 없을 때).

  transcribe()          faster-whisper로 한국어 STT (세그먼트 + 신뢰도)
  diarize()             pyannote.audio로 화자 분리
  assign_speakers()     STT 세그먼트마다 가장 많이 겹치는 화자를 붙인다 (순수 함수)

transcribe/diarize는 `pip install -e .[audio]`와 Hugging Face 토큰(HF_TOKEN)이 필요하다.
CPU에서는 Whisper medium(int8), Colab에서는 large-v3를 권장한다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SttSegment:
    start: float
    end: float
    text: str
    confidence: float


@dataclass
class SpeakerSpan:
    start: float
    end: float
    speaker: str


def transcribe(
    audio_path: str | Path,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
) -> list[SttSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - 선택 의존성
        raise RuntimeError("음성 전사에는 faster-whisper가 필요합니다: pip install -e .[audio]") from exc

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        str(audio_path), language="ko", vad_filter=True, initial_prompt=initial_prompt
    )
    out = []
    for s in segments:
        # 평균 로그확률을 0~1 신뢰도로 바꾼다 (avg_logprob 0 → 1, -1 → 약 0.37)
        out.append(SttSegment(s.start, s.end, s.text.strip(), round(math.exp(min(0.0, s.avg_logprob)), 3)))
    return out


def diarize(audio_path: str | Path, num_speakers: int | None = 2, hf_token: str | None = None) -> list[SpeakerSpan]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - 선택 의존성
        raise RuntimeError("화자 분리에는 pyannote.audio가 필요합니다: pip install -e .[audio]") from exc

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("화자 분리 모델을 받으려면 HF_TOKEN 환경변수가 필요합니다 (.env.example 참고)")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    result = pipeline(str(audio_path), num_speakers=num_speakers)
    return [SpeakerSpan(turn.start, turn.end, speaker) for turn, _, speaker in result.itertracks(yield_label=True)]


def assign_speakers(segments: list[SttSegment], spans: list[SpeakerSpan]) -> list[dict]:
    """STT 세그먼트에 겹치는 시간이 가장 긴 화자를 붙여 전사본 turns 형식으로 돌려준다.

    겹치는 화자가 없으면 시간상 가장 가까운 화자를 붙인다.
    같은 화자의 연속 세그먼트는 합치지 않는다 (발화 단위 감정 변화를 보존).
    """
    turns = []
    for seg in segments:
        best, best_overlap = None, 0.0
        for span in spans:
            overlap = min(seg.end, span.end) - max(seg.start, span.start)
            if overlap > best_overlap:
                best, best_overlap = span.speaker, overlap
        if best is None and spans:
            mid = (seg.start + seg.end) / 2
            best = min(spans, key=lambda s: min(abs(mid - s.start), abs(mid - s.end))).speaker
        turns.append(
            {"speaker": best or "S0", "start": seg.start, "end": seg.end, "text": seg.text, "stt_conf": seg.confidence}
        )
    return turns


def audio_to_transcript(audio_path: str | Path, model_size: str = "medium", device: str = "cpu") -> dict:
    """음성 파일 하나를 전사본 dict로 만든다 (ingest.call_transcript 형식)."""
    path = Path(audio_path)
    segments = transcribe(path, model_size=model_size, device=device)
    spans = diarize(path)
    return {"call_id": path.stem, "audio": str(path), "turns": assign_speakers(segments, spans), "meta": {}}
