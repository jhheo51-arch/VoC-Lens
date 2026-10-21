"""영상 내용 구간 분할 (TextTiling 방식 기준선).

1. 문장마다 글자 2-gram 빈도 벡터를 만든다 (임베딩 모델 없이 동작).
2. 경계 후보 위치마다 앞 block_size 문장과 뒤 block_size 문장의 코사인 유사도를 잰다.
3. 유사도 곡선이 양옆보다 깊게 패인 곳(depth ≥ depth_threshold)을 경계로 고른다.
4. "총평", "장점은" 같은 전환 표현으로 시작하는 문장 앞도 경계로 둔다.
5. 구간마다 키워드로 이름(도입·언박싱·사용·장단점·총평)을 붙인다.
   마지막 구간에 총평 키워드가 있으면 총평으로 본다.
2주차 이후 문장 임베딩으로 바꿔 비교한다.

PositionalSegmenter(v0.2)는 경계는 그대로 두고, 구간 이름을 붙일 때 영상 안 시간 위치(도입은 앞, 총평은 뒤)와
제품군을 함께 보고 같은 이름의 이웃 구간을 합친다. 규칙은 configs/video.yaml의 position_segmentation 주석에 있다.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from voclens.config import load_yaml


def _bigram_counts(text: str) -> Counter:
    s = re.sub(r"\s+", "", text)
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = np.sqrt(sum(v * v for v in a.values()))
    nb = np.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


@dataclass
class Segment:
    index: int
    start: int          # 문장 인덱스 (포함)
    end: int            # 문장 인덱스 (제외)
    label: str


class TextTilingSegmenter:
    name = "texttiling-bigram-v0.1"

    def __init__(self, config: dict | None = None):
        cfg = config or load_yaml("video.yaml")
        seg = cfg["segmentation"]
        self.block = seg["block_size"]
        self.depth_threshold = seg["depth_threshold"]
        self.min_sentences = seg["min_sentences"]
        self.cues = tuple(seg["cue_phrases"])
        self.labels = cfg["segment_labels"]
        self.default_label = cfg["default_segment_label"]

    def gap_scores(self, sentences: list[str]) -> list[float]:
        """gap i = 문장 i-1과 문장 i 사이의 유사도 (i = 1..n-1)."""
        counts = [_bigram_counts(s) for s in sentences]
        scores = []
        for i in range(1, len(sentences)):
            left = sum((counts[j] for j in range(max(0, i - self.block), i)), Counter())
            right = sum((counts[j] for j in range(i, min(len(sentences), i + self.block))), Counter())
            scores.append(_cosine(left, right))
        return scores

    def boundaries(self, sentences: list[str]) -> list[int]:
        n = len(sentences)
        if n < 2 * self.min_sentences:
            return []
        scores = self.gap_scores(sentences)
        candidates = []
        for k, s in enumerate(scores):
            # 양옆으로 값이 오르는 동안만 올라가서 만나는 국소 봉우리 (TextTiling 원 방식)
            left = k
            while left > 0 and scores[left - 1] >= scores[left]:
                left -= 1
            right = k
            while right < len(scores) - 1 and scores[right + 1] >= scores[right]:
                right += 1
            depth = (scores[left] - s) + (scores[right] - s)
            if depth >= self.depth_threshold:
                candidates.append((depth, k + 1))
        cue_starts = [i for i, s in enumerate(sentences) if i > 0 and s.lstrip().startswith(self.cues)]
        # 전환 표현 경계를 먼저 넣고, 깊이가 큰 순서로 최소 길이를 지키며 추가한다
        chosen: list[int] = []
        for _, pos in [(float("inf"), c) for c in cue_starts] + sorted(candidates, reverse=True):
            if pos < self.min_sentences or n - pos < self.min_sentences:
                continue
            if all(abs(pos - c) >= self.min_sentences for c in chosen):
                chosen.append(pos)
        return sorted(chosen)

    def _label(self, sentences: list[str], is_last: bool, is_first: bool) -> str:
        text = " ".join(sentences)
        counts = {name: sum(text.count(k) for k in kws) for name, kws in self.labels.items()}
        if is_last and counts.get("총평", 0) > 0:
            return "총평"
        if is_first and counts.get("도입", 0) > 0:
            return "도입"
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else self.default_label

    def segment(self, sentences: list[str], positions: list[float] | None = None, meta: dict | None = None) -> list[Segment]:
        """positions·meta는 PositionalSegmenter와 인터페이스를 맞추려고 받기만 한다."""
        if not sentences:
            return []
        cuts = [0, *self.boundaries(sentences), len(sentences)]
        segments = []
        for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
            label = self._label(sentences[a:b], is_last=(b == len(sentences)), is_first=(a == 0))
            segments.append(Segment(i, a, b, label))
        return segments


def product_group(meta: dict | None, groups: dict) -> str | None:
    """제목·태그에 제품군 키워드가 있으면 그 제품군 이름 (설명란은 링크·광고가 섞여 보지 않는다)."""
    if not meta:
        return None
    text = " ".join([meta.get("title") or "", " ".join(meta.get("tags") or [])])
    for name, spec in groups.items():
        if any(k in text for k in spec["keywords"]):
            return name
    return None


class PositionalSegmenter(TextTilingSegmenter):
    """TextTiling 경계 + 시간 위치·제품군을 본 구간 이름 + 같은 이름 병합 (v0.2)."""

    name = "texttiling-bigram-position-v0.2"

    def __init__(self, config: dict | None = None):
        cfg = config or load_yaml("video.yaml")
        super().__init__(cfg)
        pos = cfg["position_segmentation"]
        self.windows = pos["label_windows"]
        self.groups = pos.get("product_groups") or {}

    def _allowed(self, label: str, start: float, mid: float, excluded: set[str]) -> bool:
        if label in excluded:
            return False
        w = self.windows.get(label, {})
        return start <= w.get("start_max", 1.0) and mid >= w.get("mid_min", 0.0)

    def _counts(self, sentences: list[str]) -> dict[str, int]:
        text = " ".join(sentences)
        return {name: sum(text.count(k) for k in kws) for name, kws in self.labels.items()}

    def segment(self, sentences: list[str], positions: list[float] | None = None, meta: dict | None = None) -> list[Segment]:
        n = len(sentences)
        if not n:
            return []
        if positions is None or len(positions) != n:
            positions = [(i + 0.5) / n for i in range(n)]
        group = product_group(meta, self.groups)
        excluded = set(self.groups[group].get("exclude_labels", [])) if group else set()
        cuts = [0, *self.boundaries(sentences), n]
        spans = list(zip(cuts, cuts[1:]))
        info = []
        for a, b in spans:
            start = 0.0 if a == 0 else positions[a]
            mid = positions[(a + b - 1) // 2] if b - a > 1 else positions[a]
            counts = {k: v for k, v in self._counts(sentences[a:b]).items() if self._allowed(k, start, mid, excluded)}
            info.append((start, mid, counts))

        labels: list[str | None] = [None] * len(spans)
        # 규칙 3: 도입은 첫머리부터 이어질 때만
        for i, (_, _, counts) in enumerate(info):
            if counts.get("도입", 0) > 0:
                labels[i] = "도입"
            else:
                break
        # 규칙 4: 마지막 구간부터 거꾸로 총평을 이어 붙인다
        for i in range(len(spans) - 1, -1, -1):
            if labels[i] is None and info[i][2].get("총평", 0) > 0:
                labels[i] = "총평"
            else:
                break
        # 나머지: 허용된 이름 가운데 키워드가 가장 많은 것 (도입은 규칙 3에서만 붙인다)
        for i, (_, _, counts) in enumerate(info):
            if labels[i] is not None:
                continue
            counts = {k: v for k, v in counts.items() if k != "도입"}
            best = max(counts, key=counts.get) if counts else None
            labels[i] = best if best and counts[best] > 0 else self.default_label

        # 규칙 6: 같은 이름의 이웃 구간 병합
        merged: list[list] = []
        for (a, b), label in zip(spans, labels):
            if merged and merged[-1][2] == label:
                merged[-1][1] = b
            else:
                merged.append([a, b, label])
        return [Segment(i, a, b, label) for i, (a, b, label) in enumerate(merged)]


def make_segmenter(method: str | None = None, config: dict | None = None):
    """설정(segmentation.method) 또는 인자로 분할기를 고른다: texttiling(기본) | position."""
    cfg = config or load_yaml("video.yaml")
    method = method or cfg["segmentation"].get("method", "texttiling")
    if method == "position":
        return PositionalSegmenter(cfg)
    if method == "texttiling":
        return TextTilingSegmenter(cfg)
    raise ValueError(f"알 수 없는 구간 분할 방식: {method} (texttiling | position)")
