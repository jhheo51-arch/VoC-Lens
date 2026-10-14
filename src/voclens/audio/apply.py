"""음성 파일이 있을 때 발화마다 음향 각성을 추정해 텍스트 감정과 융합한다 (상담·영상 공통)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voclens.audio import acoustic
from voclens.audio.ser import AcousticArousalBaseline
from voclens.fusion.fuse import fuse_unit


@dataclass
class Span:
    start: float
    end: float
    text: str
    speaker: str


def apply_audio(units, spans: list[Span], path: Path | None, ser: AcousticArousalBaseline, settings: dict) -> dict:
    """units와 spans는 같은 순서·길이여야 한다. 반환값은 세션 메타데이터에 넣을 정보."""
    if not path or not path.exists():
        return {"audio_used": False}
    if path.suffix.lower() != ".wav":
        return {"audio_used": False, "audio_note": "WAV만 직접 읽습니다. ffmpeg로 16kHz 모노 WAV로 바꿔 주세요."}

    signal, sr = acoustic.read_wav(path)
    feats = [acoustic.segment_features(signal, sr, s.start, s.end, s.text) for s in spans]
    zs = acoustic.normalize_by_speaker(
        feats, [s.speaker for s in spans], [s.start for s in spans], settings["call"]["acoustic_baseline_sec"]
    )
    snr = acoustic.estimate_snr_db(signal, sr)
    predictions = ser.predict(zs, [f.reliable for f in feats], snr_db=snr)
    for unit, pred in zip(units, predictions):
        fuse_unit(unit, pred.valence, pred.arousal, pred.confidence, settings["call"]["fusion"])
    return {"audio_used": True, "snr_db": round(snr, 1), "sample_rate": sr}
