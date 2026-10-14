"""음성 감정 인식 (Speech Emotion Recognition).

AcousticArousalBaseline: 화자 기준 z-score 음향 특징으로 각성(Arousal)만 추정하는 규칙 기준선.
  목소리는 "얼마나 흥분했는가"는 잘 드러내지만 "좋은가 나쁜가"는 잘 구분하지 못한다.
  그래서 기준선은 Valence를 내지 않고(None), 융합에서 텍스트 Valence를 쓴다.

3주차 Colab 작업에서 emotion2vec 임베딩 + 한국어 감정 음성 분류기로 바꾼다.
predict()의 입력·출력 형식은 그대로 둔다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score 특징별 가중치. 높은 피치·큰 변동·큰 에너지·빠른 말 속도 = 높은 각성
AROUSAL_WEIGHTS = {"pitch_mean": 0.35, "pitch_std": 0.20, "energy_db": 0.35, "speech_rate": 0.20, "pause_ratio": -0.15}
# 화자 평소 상태(z=0)의 각성. 상담 대화의 평상시 각성을 중간보다 약간 낮게 둔다
BASELINE_AROUSAL = 0.4


@dataclass
class AudioEmotion:
    arousal: float | None
    valence: float | None
    confidence: float


class AcousticArousalBaseline:
    name = "acoustic-arousal-v0.1"

    def predict_one(self, z: dict[str, float], reliable: bool, snr_db: float | None = None) -> AudioEmotion:
        if not z:
            return AudioEmotion(None, None, 0.0)
        score = sum(w * z.get(k, 0.0) for k, w in AROUSAL_WEIGHTS.items())
        logit = math.log(BASELINE_AROUSAL / (1 - BASELINE_AROUSAL))
        arousal = 1 / (1 + math.exp(-(logit + score)))
        confidence = 0.6 if reliable else 0.1
        if snr_db is not None:
            confidence *= min(1.0, max(0.1, (snr_db - 5) / 15))
        return AudioEmotion(round(arousal, 4), None, round(confidence, 3))

    def predict(self, zs: list[dict[str, float]], reliable: list[bool], snr_db: float | None = None) -> list[AudioEmotion]:
        return [self.predict_one(z, r, snr_db) for z, r in zip(zs, reliable)]
