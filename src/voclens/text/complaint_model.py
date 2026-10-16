"""학습한 불만 유형 분류기 (2단계). KeywordTaxonomyClassifier와 같은 predict(units, domain) 인터페이스.

모델은 절(앞 절을 문맥으로 붙임)마다 불만 유형별 확률을 낸다. 합치는 방식
  union  규칙이 잡은 유형 ∪ 모델이 threshold 이상으로 낸 유형 (놓친 불만을 보충)
  model  모델 결과만 (규칙은 칭찬·분류 불가·속성 판정에만 쓴다)
칭찬·분류 불가, 속성, 명시적 불만의 Valence 상한은 규칙 분류기와 같은 방식으로 처리한다.

    python scripts/train_complaint.py      # → models/complaint-weak-small
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voclens.schema import Unit
from voclens.text.preprocess import preprocess
from voclens.text.taxonomy import COMPLAINT_VALENCE_CAP, KeywordTaxonomyClassifier, normalize_domain


def model_input(text: str, prev: str = "") -> str:
    """앞 절을 문맥으로 붙인다 ("블루를 샀는데" + "블랙이 왔어요")."""
    return f"{preprocess(prev).text} [SEP] {preprocess(text).text}" if prev else preprocess(text).text


class TransformerComplaintClassifier(KeywordTaxonomyClassifier):
    def __init__(self, model_dir: str | Path, threshold: float = 0.5, mode: str = "union", batch_size: int = 64,
                 max_length: int = 96, **kwargs):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        super().__init__(**kwargs)
        self.model_dir = Path(model_dir)
        self.name = f"transformer:{self.model_dir.name}:{mode}@{threshold}"
        self.labels = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.threshold, self.mode = threshold, mode
        self.batch_size, self.max_length = batch_size, max_length
        self._cache: dict[str, np.ndarray] = {}

    def probs(self, inputs: list[str]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(inputs), self.batch_size):
            enc = self.tokenizer(inputs[i : i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            # 0번 "불만 없음"을 포함한 softmax 분류. 유형별 확률을 그대로 임계치와 비교한다
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(self.labels)))

    def prepare(self, units: list[Unit]) -> None:
        """여러 세션의 절을 한 번에 추론해 둔다 (파이프라인이 세션마다 predict를 불러도 빠르게)."""
        todo, inputs, prev, prev_session = [], [], "", None
        for u in units:
            text = u.clean_text or u.text
            if u.unit_id not in self._cache:
                todo.append(u.unit_id)
                inputs.append(model_input(text, prev if u.session_id == prev_session else ""))
            prev, prev_session = text, u.session_id
        for uid, p in zip(todo, self.probs(inputs)):
            self._cache[uid] = p

    def model_labels(self, unit: Unit, domain: str | None) -> list[str]:
        p = self._cache[unit.unit_id]
        return [l for l, v in zip(self.labels, p) if v >= self.threshold and l in self.taxonomy.subcategories
                and self._in_domain(l, domain)]

    def predict(self, units: list[Unit], domain: str | None = None) -> list[Unit]:
        self.prepare(units)
        valences = {u.unit_id: u.valence for u in units}          # 규칙 분류기가 Valence를 낮추기 전 값
        super().predict(units, domain)
        tx, domain = self.taxonomy, normalize_domain(domain)
        for unit in units:
            rules = [l for l in unit.complaint_categories if tx.is_complaint(l)]
            found = self.model_labels(unit, domain)
            merged = list(dict.fromkeys(rules + found)) if self.mode == "union" else found
            if merged == rules:
                continue
            valence = valences[unit.unit_id] or 0.0
            if merged:
                unit.complaint_categories = merged
                unit.confidence["taxonomy"] = 1.0 if set(merged) & set(rules) else 0.8
                unit.valence = min(valence, COMPLAINT_VALENCE_CAP)
                unit.aspect = tx.subcategories[merged[0]]["aspect"]
            else:   # model 모드에서 규칙만 잡은 불만을 버린 경우
                unit.valence = valence
                unit.complaint_categories = ([tx.praise_label] if valence >= tx.praise_min_valence
                                             else [tx.unknown_label] if valence <= tx.unknown_max_valence else [])
                unit.confidence["taxonomy"] = 0.3 if unit.complaint_categories else 0.0
                unit.aspect = self.find_aspect(unit.clean_text or unit.text)
            unit.aspect_sentiment = unit.valence if unit.aspect else None
        return units
