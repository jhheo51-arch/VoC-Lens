"""VoC 불만 유형 분류.

구축 단계 (docs/analysis_design.md 1.3)
  1단계  KeywordTaxonomyClassifier  — 분류 기준의 complaint/topic 정규식 (기준선)
  1단계  EmbeddingTaxonomyClassifier — 시드 문장 임베딩과의 코사인 유사도 (Zero-shot)
  2단계  uncertainty_ranking()       — 능동 학습: 모델이 헷갈리는 절부터 라벨링
  3단계  파인튜닝 분류기 (2주차, Colab)

분류 결과는 소분류 이름(예: "배달 지연")으로 저장하고, 대분류는 category_of()로 찾는다.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from voclens.config import load_yaml
from voclens.schema import Unit

# 불만 표현이 있으면 감정 사전이 놓쳐도 이 값 이하의 Valence로 본다 ("1시간 걸렸어요")
COMPLAINT_VALENCE_CAP = -0.3

_DOMAIN_ALIASES = {
    "delivery": "delivery", "배달": "delivery", "food": "delivery", "음식": "delivery",
    "product": "product", "제품": "product", "가전": "product", "전자": "product",
}


def normalize_domain(value: str | None) -> str | None:
    """메타데이터의 도메인 표기(배달, product 등)를 분류 기준의 도메인 이름으로. 모르면 None(전체 사용)."""
    return _DOMAIN_ALIASES.get(str(value).strip().lower()) if value else None


@dataclass
class Taxonomy:
    subcategories: dict[str, dict]
    aspect_keywords: dict[str, list[str]]
    praise_label: str
    unknown_label: str
    praise_min_valence: float
    unknown_max_valence: float

    @classmethod
    def load(cls, data: dict | None = None) -> Taxonomy:
        d = data or load_yaml("taxonomy.yaml")
        return cls(
            subcategories=d["subcategories"],
            aspect_keywords=d.get("aspect_keywords", {}),
            praise_label=d["praise_label"],
            unknown_label=d["unknown_label"],
            praise_min_valence=float(d["praise_min_valence"]),
            unknown_max_valence=float(d["unknown_max_valence"]),
        )

    def category_of(self, label: str) -> str:
        sub = self.subcategories.get(label)
        return sub["category"] if sub else label

    def difficulty_of(self, label: str) -> int:
        sub = self.subcategories.get(label)
        return int(sub.get("difficulty", 2)) if sub else 2

    def is_complaint(self, label: str) -> bool:
        return label in self.subcategories


def _compile(patterns: list[str]) -> re.Pattern | None:
    return re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None


class KeywordTaxonomyClassifier:
    name = "keyword-v0.1"

    def __init__(self, taxonomy: Taxonomy | None = None):
        self.taxonomy = taxonomy or Taxonomy.load()
        self._complaint = {k: _compile(v.get("complaint", [])) for k, v in self.taxonomy.subcategories.items()}
        self._topic = {k: _compile(v.get("topic", [])) for k, v in self.taxonomy.subcategories.items()}
        self._aspect = {a: _compile(kws) for a, kws in self.taxonomy.aspect_keywords.items()}

    def _in_domain(self, label: str, domain: str | None) -> bool:
        domains = self.taxonomy.subcategories[label].get("domains")
        return domain is None or not domains or domain in domains

    def classify(self, text: str, valence: float, domain: str | None = None) -> list[tuple[str, float]]:
        """(소분류, 강도) 목록. 강도 1.0 = 불만 표현, 0.5 = 주제 + 부정 감정.

        domain을 주면 다른 도메인 전용 소분류는 쓰지 않는다 (제품 영상의 "충전이 오래 걸려"가 배달 지연이 되지 않게).
        """
        hits: dict[str, float] = {}
        for label, pattern in self._complaint.items():
            if pattern and self._in_domain(label, domain) and pattern.search(text):
                hits[label] = 1.0
        if valence < 0:
            for label, pattern in self._topic.items():
                if label not in hits and pattern and self._in_domain(label, domain) and pattern.search(text):
                    hits[label] = 0.5
        return sorted(hits.items(), key=lambda kv: -kv[1])

    def explain(self, text: str, valence: float, domain: str | None = None) -> str:
        """불만 유형 판단 근거. 예: "누락·오배송 ← '빠져 있'(불만 표현)", "가격 ← '가격'(주제 + 부정 감성)"."""
        domain = normalize_domain(domain)
        parts = []
        for label, pattern in self._complaint.items():
            if pattern and self._in_domain(label, domain) and (m := pattern.search(text)):
                parts.append(f"{label} ← '{m.group(0)}'(불만 표현)")
        if valence < 0:
            found = {p.split(" ←")[0] for p in parts}
            for label, pattern in self._topic.items():
                if label not in found and pattern and self._in_domain(label, domain) and (m := pattern.search(text)):
                    parts.append(f"{label} ← '{m.group(0)}'(주제 + 부정 감성)")
        return ", ".join(parts)

    def find_aspect(self, text: str) -> str | None:
        for aspect, pattern in self._aspect.items():
            if pattern and pattern.search(text):
                return aspect
        return None

    def predict(self, units: list[Unit], domain: str | None = None) -> list[Unit]:
        tx = self.taxonomy
        domain = normalize_domain(domain)
        for unit in units:
            text = unit.clean_text or unit.text
            valence = unit.valence or 0.0
            hits = self.classify(text, valence, domain)

            if hits:
                unit.complaint_categories = [label for label, _ in hits]
                unit.confidence["taxonomy"] = hits[0][1]
                if hits[0][1] >= 1.0 and valence > COMPLAINT_VALENCE_CAP:
                    unit.valence = COMPLAINT_VALENCE_CAP
                unit.aspect = tx.subcategories[hits[0][0]]["aspect"]
            else:
                if valence >= tx.praise_min_valence:
                    unit.complaint_categories = [tx.praise_label]
                elif valence <= tx.unknown_max_valence:
                    unit.complaint_categories = [tx.unknown_label]
                unit.confidence["taxonomy"] = 0.3 if unit.complaint_categories else 0.0
                unit.aspect = self.find_aspect(text)

            unit.aspect_sentiment = unit.valence if unit.aspect else None
        return units


Encoder = Callable[[list[str]], np.ndarray]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class EmbeddingTaxonomyClassifier:
    """시드 문장 임베딩의 평균(중심)과 코사인 유사도로 분류하는 Zero-shot 분류기.

    encode는 문장 목록을 (n, d) 배열로 바꾸는 함수다. 실제로는
    sentence_transformer_encoder()를 쓰고, 테스트에서는 가짜 인코더를 넣는다.
    """

    name = "embedding-zeroshot-v0.1"

    def __init__(self, encode: Encoder, taxonomy: Taxonomy | None = None, threshold: float = 0.5):
        self.taxonomy = taxonomy or Taxonomy.load()
        self.encode = encode
        self.threshold = threshold
        self.labels = [k for k, v in self.taxonomy.subcategories.items() if v.get("seeds")]
        centroids = [
            _normalize(encode(self.taxonomy.subcategories[k]["seeds"])).mean(axis=0) for k in self.labels
        ]
        self.centroids = _normalize(np.vstack(centroids))

    def similarities(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.encode(texts)) @ self.centroids.T

    def classify(self, texts: list[str]) -> list[tuple[str | None, float, float]]:
        """(소분류 또는 None, 최고 유사도, 1위-2위 차이) 목록."""
        sims = self.similarities(texts)
        out = []
        for row in sims:
            order = np.argsort(row)[::-1]
            best = float(row[order[0]])
            margin = best - float(row[order[1]]) if len(order) > 1 else best
            label = self.labels[order[0]] if best >= self.threshold else None
            out.append((label, best, margin))
        return out


def uncertainty_ranking(results: list[tuple[str | None, float, float]]) -> list[int]:
    """능동 학습용: 1·2위 차이가 작은(헷갈리는) 순서로 인덱스를 돌려준다."""
    return sorted(range(len(results)), key=lambda i: results[i][2])


def sentence_transformer_encoder(model_name: str = "jhgan/ko-sroberta-multitask") -> Encoder:
    """sentence-transformers 인코더. `pip install -e .[nlp]` 필요."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: np.asarray(model.encode(texts, convert_to_numpy=True))
