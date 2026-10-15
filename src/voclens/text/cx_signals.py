"""CX 신호 탐지 (사전 기반 기준선).

재주문·이탈·추천 의사, 반복 문의, 대기, 해결, 사과 표현을 찾는다.
좌절(고객 노력) 지수와 NPS 성향 추정의 재료가 된다.
"""

from __future__ import annotations

import re

from voclens.config import load_yaml
from voclens.schema import Unit


SIGNAL_LABELS = {
    "repurchase_intent": "재주문 의사",
    "churn_intent": "이탈 의사",
    "recommend_intent": "추천 의사",
    "not_recommend": "비추천",
    "repeat_contact": "반복 문의",
    "waiting": "대기",
    "resolution": "해결",
    "apology_received": "사과 받음",
    "tone_mismatch": "말–톤 불일치",
}


class CxSignalDetector:
    name = "cx-signals-v0.1"

    def __init__(self, lexicon: dict | None = None):
        data = lexicon or load_yaml("lexicons/cx_signals.yaml")
        self.signals = {
            name: (
                re.compile("|".join(spec["patterns"])),
                re.compile("|".join(spec["exclude"])) if spec.get("exclude") else None,
            )
            for name, spec in data["signals"].items()
        }

    def detect(self, text: str) -> dict[str, bool]:
        found = {}
        for name, (pattern, exclude) in self.signals.items():
            if pattern.search(text) and not (exclude and exclude.search(text)):
                found[name] = True
        return found

    def explain(self, text: str) -> str:
        """CX 신호 판단 근거. 예: "repurchase_intent ← '또 시킬'"."""
        parts = []
        for name, (pattern, exclude) in self.signals.items():
            m = pattern.search(text)
            if m and not (exclude and exclude.search(text)):
                parts.append(f"{SIGNAL_LABELS.get(name, name)} ← '{m.group(0)}'")
        return ", ".join(parts)

    def predict(self, units: list[Unit]) -> list[Unit]:
        for unit in units:
            unit.cx_signals = self.detect(unit.clean_text or unit.text)
        return units
