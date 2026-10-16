"""학습한 감정 분류 모델 (2단계). LexiconEmotionModel과 같은 인터페이스(score, predict)를 따른다.

KOTE 44감정을 VoC Lens 6감정(기쁨·슬픔·분노·혐오·불안·놀람)으로 묶어 다중 라벨(시그모이드)로 학습한다.
중립은 "어느 감정도 아닐 확률" = Π(1 − p)로 두고, 7감정을 정규화해 Circumplex 좌표로 Valence·Arousal을 낸다.
그래서 분류 기준의 Valence 임계치(칭찬 0.25, 분류 불가 −0.2, 주제 표현은 V < 0)를 그대로 쓸 수 있다.

    pip install -e ".[train]"
    python scripts/train_emotion.py            # → models/emotion-kote-small
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voclens.config import load_yaml
from voclens.schema import EMOTIONS, Unit
from voclens.text.emotion import EmotionResult
from voclens.text.preprocess import preprocess

GROUPS = ("기쁨", "슬픔", "분노", "혐오", "불안", "놀람")   # 모델 출력 순서
HIT_THRESHOLD = 0.5   # 어떤 감정 확률이 이 값을 넘으면 "감정 표현이 있다"로 본다 (사전의 hit에 대응)


class TransformerEmotionModel:
    def __init__(self, model_dir: str | Path, batch_size: int = 64, max_length: int = 96, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.name = f"transformer:{self.model_dir.name}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size, self.max_length = batch_size, max_length
        self.circumplex = {e: tuple(v) for e, v in load_yaml("lexicons/emotion.yaml")["circumplex"].items()}

    def group_probs(self, texts: list[str]) -> np.ndarray:
        """(문장 수, 6) 시그모이드 확률."""
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [preprocess(t).text for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(torch.sigmoid(self.model(**enc).logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(GROUPS)))

    def _result(self, p: np.ndarray) -> EmotionResult:
        scores = dict(zip(GROUPS, p.tolist()))
        scores["중립"] = float(np.prod(1 - p))
        total = sum(scores.values()) or 1.0
        probs = {e: scores.get(e, 0.0) / total for e in EMOTIONS}
        valence = sum(probs[e] * self.circumplex[e][0] for e in EMOTIONS)
        arousal = sum(probs[e] * self.circumplex[e][1] for e in EMOTIONS)
        top = float(p.max()) if len(p) else 0.0
        return EmotionResult(probs=probs, valence=valence, arousal=arousal, hit_weight=top if top >= HIT_THRESHOLD else 0.0)

    def score_many(self, texts: list[str]) -> list[EmotionResult]:
        return [self._result(p) for p in self.group_probs(texts)]

    def score(self, text: str) -> EmotionResult:
        return self.score_many([text])[0]

    def explain(self, text: str) -> str:
        p = self.group_probs([text])[0]
        return ", ".join(f"{g} {v:.2f}" for g, v in sorted(zip(GROUPS, p), key=lambda kv: -kv[1]) if v >= 0.3) or "뚜렷한 감정 없음"

    def predict(self, units: list[Unit]) -> list[Unit]:
        results = self.score_many([u.clean_text or u.text for u in units])
        for unit, res in zip(units, results):
            unit.emotion_probs = {e: round(p, 4) for e, p in res.probs.items()}
            unit.valence = round(res.valence, 4)
            unit.arousal = round(res.arousal, 4)
            unit.confidence["text"] = round(min(1.0, 0.2 + res.hit_weight), 3)
        return units
