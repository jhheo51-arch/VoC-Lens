"""영상 유형 분류 (키워드 점수 기준선).

제목·태그는 가중치 2, 설명·자막 앞부분은 1로 키워드 등장 수를 센다.
가장 높은 유형을 고르고, 1·2위 차이로 신뢰도를 낸다. 점수가 0이면 "기타".
2주차 이후 임베딩 Zero-shot과 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from voicecx.config import load_yaml

TITLE_WEIGHT = 2.0
BODY_WEIGHT = 1.0
HEAD_SENTENCES = 10


@dataclass
class VideoType:
    type: str
    label: str
    confidence: float
    scores: dict[str, float]


class VideoTypeClassifier:
    name = "video-type-keyword-v0.1"

    def __init__(self, config: dict | None = None):
        cfg = config or load_yaml("video.yaml")
        self.types = cfg["types"]
        self.default = cfg["default_type"]
        self.other_label = cfg["other_label"]

    def classify(self, meta: dict, sentences: list[str]) -> VideoType:
        title = " ".join([meta.get("title") or "", " ".join(meta.get("tags") or [])])
        body = " ".join([meta.get("description") or "", " ".join(sentences[:HEAD_SENTENCES])])
        scores = {}
        for name, spec in self.types.items():
            scores[name] = sum(
                TITLE_WEIGHT * title.count(kw) + BODY_WEIGHT * body.count(kw) for kw in spec["keywords"]
            )
        ranked = sorted(scores, key=scores.get, reverse=True)
        top = scores[ranked[0]]
        if top == 0:
            return VideoType(self.default, self.other_label, 0.0, scores)
        second = scores[ranked[1]] if len(ranked) > 1 else 0.0
        return VideoType(ranked[0], self.types[ranked[0]]["label"], round((top - second) / top, 3), scores)
