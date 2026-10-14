"""음향 + 텍스트 융합 (late fusion).

차원마다 가중치를 다르게 둔다 (docs/analysis_design.md 5장).
  각성(Arousal):        음향 0.7 / 텍스트 0.3
  긍정·부정(Valence):   음향 0.3 / 텍스트 0.7
실제 가중치 = 기본 가중치 × 각 모달의 신뢰도. 한쪽 값이 없으면 다른 쪽만 쓴다.
4주차에 KEMDy로 스태킹 모델을 학습해 기본 가중치를 데이터로 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fused:
    value: float | None
    confidence: float
    audio_weight: float


def fuse_dimension(
    text_value: float | None,
    text_conf: float,
    audio_value: float | None,
    audio_conf: float,
    weights: tuple[float, float],
) -> Fused:
    w_audio = weights[0] * audio_conf if audio_value is not None else 0.0
    w_text = weights[1] * text_conf if text_value is not None else 0.0
    total = w_audio + w_text
    if total == 0:
        value = text_value if text_value is not None else audio_value
        return Fused(value, 0.0, 0.0)
    value = (w_audio * (audio_value or 0.0) + w_text * (text_value or 0.0)) / total
    # 융합 신뢰도: 기본 가중치로 가중한 두 신뢰도의 합 (둘 다 확실하면 1에 가까움)
    confidence = min(1.0, w_audio + w_text)
    return Fused(round(value, 4), round(confidence, 3), round(w_audio / total, 3))


def fuse_unit(unit, audio_valence, audio_arousal, audio_conf: float, weights: dict) -> None:
    """Unit의 텍스트 기반 valence/arousal을 음향 결과와 융합해 덮어쓴다. 원래 값은 confidence에 남긴다."""
    text_conf = unit.confidence.get("text", 0.2) * unit.confidence.get("stt", 1.0)
    arousal = fuse_dimension(unit.arousal, text_conf, audio_arousal, audio_conf, tuple(weights["arousal"]))
    valence = fuse_dimension(unit.valence, text_conf, audio_valence, audio_conf, tuple(weights["valence"]))
    unit.confidence["audio"] = audio_conf
    if unit.arousal is not None:
        unit.confidence["text_arousal"] = unit.arousal
    if audio_arousal is not None:
        unit.confidence["audio_arousal"] = audio_arousal
    unit.confidence["fusion_audio_weight_arousal"] = arousal.audio_weight
    unit.confidence["fusion"] = arousal.confidence
    unit.arousal = arousal.value
    unit.valence = valence.value
