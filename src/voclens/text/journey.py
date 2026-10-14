"""고객 여정 단계 태깅 (규칙 기반 기준선).

불만 유형이 있으면 분류 기준의 stage를 쓰고, 없으면 journey_stages.yaml의 키워드로 추정한다.
"""

from __future__ import annotations

import re

from voclens.config import load_yaml
from voclens.schema import Unit
from voclens.text.taxonomy import Taxonomy


class JourneyTagger:
    name = "journey-keyword-v0.1"

    def __init__(self, stages: dict | None = None, taxonomy: Taxonomy | None = None):
        data = stages or load_yaml("journey_stages.yaml")
        self.stage_names = [s["name"] for s in data["stages"]]
        self._patterns = [(s["name"], re.compile("|".join(s["keywords"]))) for s in data["stages"]]
        self.taxonomy = taxonomy or Taxonomy.load()

    def tag(self, text: str, categories: list[str]) -> str | None:
        for label in categories:
            sub = self.taxonomy.subcategories.get(label)
            if sub and sub.get("stage"):
                return sub["stage"]
        for name, pattern in self._patterns:
            if pattern.search(text):
                return name
        return None

    def predict(self, units: list[Unit]) -> list[Unit]:
        for unit in units:
            unit.journey_stage = self.tag(unit.clean_text or unit.text, unit.complaint_categories)
        return units
