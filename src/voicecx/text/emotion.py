"""사전 기반 감정 분류기 (기준선).

절마다 7감정 확률을 구하고, Circumplex 좌표로 Valence(-1~1)와 Arousal(0~1)을 계산한다.
2주차에 파인튜닝 모델로 바꿔도 predict() 인터페이스는 그대로 둔다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voicecx.config import load_yaml
from voicecx.schema import EMOTIONS, Unit

POSITIVE = {"기쁨"}

# "안 좋아요", "못 먹겠", "덜 익"
_NEG_BEFORE = re.compile(r"(^|\s)(안|못|덜)\s?$")
# "좋지 않", "만족스럽지 않", "추천 안 함", "추천은 못"
_NEG_AFTER = re.compile(r"^[가-힣]{0,3}(지\s?않|진\s?않|지는\s?않|지\s?못)|^[가-힣]{0,2}\s(안|못)(\s|$|해|함|하)")


@dataclass
class EmotionResult:
    probs: dict[str, float]
    valence: float
    arousal: float
    hit_weight: float


class LexiconEmotionModel:
    name = "lexicon-v0.1"

    def __init__(self, lexicon: dict | None = None):
        lex = lexicon or load_yaml("lexicons/emotion.yaml")
        self.patterns = {
            emotion: [(re.compile(p), float(w)) for p, w in entries]
            for emotion, entries in lex["emotions"].items()
        }
        self.intensifier = re.compile(r"(" + "|".join(lex["intensifiers"]) + r")\s?$")
        self.flip_positive_to = (lex["flip_positive_to"][0], float(lex["flip_positive_to"][1]))
        self.flip_negative_to = (lex["flip_negative_to"][0], float(lex["flip_negative_to"][1]))
        self.neutral_score = float(lex["neutral_score"])
        self.circumplex = {e: tuple(v) for e, v in lex["circumplex"].items()}

    def score(self, text: str) -> EmotionResult:
        scores = {e: 0.0 for e in EMOTIONS}
        hit_weight = 0.0

        for emotion, entries in self.patterns.items():
            for pattern, weight in entries:
                for m in pattern.finditer(text):
                    before = text[max(0, m.start() - 4) : m.start()]
                    after = text[m.end() : m.end() + 8]
                    w = weight * (1.5 if self.intensifier.search(before) else 1.0)
                    negated = bool(_NEG_BEFORE.search(before) or _NEG_AFTER.search(after))
                    if negated:
                        target, ratio = self.flip_positive_to if emotion in POSITIVE else self.flip_negative_to
                        scores[target] += w * ratio
                    else:
                        scores[emotion] += w
                    hit_weight += w

        scores["중립"] += self.neutral_score
        total = sum(scores.values())
        probs = {e: s / total for e, s in scores.items()}
        valence = sum(p * self.circumplex[e][0] for e, p in probs.items())
        arousal = sum(p * self.circumplex[e][1] for e, p in probs.items())
        return EmotionResult(probs=probs, valence=valence, arousal=arousal, hit_weight=hit_weight)

    def predict(self, units: list[Unit]) -> list[Unit]:
        for unit in units:
            result = self.score(unit.clean_text or unit.text)
            unit.emotion_probs = {e: round(p, 4) for e, p in result.probs.items()}
            unit.valence = round(result.valence, 4)
            unit.arousal = round(result.arousal, 4)
            # 감정 표현이 하나도 없으면 신뢰도를 낮게 둔다
            unit.confidence["text"] = round(min(1.0, 0.2 + result.hit_weight / 1.5), 3)
        return units
