"""CX 신호 탐지 (사전 기반 기준선).

재주문·이탈·추천 의사, 반복 문의, 대기, 해결, 사과 표현을 찾는다.
좌절(고객 노력) 지수와 NPS 성향 추정의 재료가 된다.
"""

from __future__ import annotations

import re

from voicecx.config import load_yaml
from voicecx.schema import Unit


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

    def predict(self, units: list[Unit]) -> list[Unit]:
        for unit in units:
            unit.cx_signals = self.detect(unit.clean_text or unit.text)
        return units
