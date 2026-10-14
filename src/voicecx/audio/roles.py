"""화자 역할 판정: 화자 분리 결과(A, B, ...)에서 누가 상담원인지 정한다.

점수 = 상담원 정형 표현 수 - 0.5 × 고객 표현 수 (발화 수로 나눔) + 첫 발화자 가산점
점수가 가장 높은 화자가 상담원, 나머지는 고객이다.
전사본에 role이 이미 있으면 그대로 쓴다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from voicecx.config import load_yaml

FIRST_SPEAKER_BONUS = 0.3


@dataclass
class RoleAssignment:
    roles: dict[str, str]          # speaker → agent | customer
    confidence: float              # 0~1, 1·2위 점수 차이 기반
    scores: dict[str, float]
    method: str                    # given | rules | single


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


class RoleAssigner:
    name = "roles-rules-v0.1"

    def __init__(self, phrases: dict | None = None):
        data = phrases or load_yaml("lexicons/call_phrases.yaml")
        self.agent = _compile(data["agent_script"])
        self.customer = _compile(data["customer_cue"])

    def assign(self, turns) -> RoleAssignment:
        speakers = list(dict.fromkeys(t.speaker for t in turns))
        given = {t.speaker: t.role for t in turns if t.role}
        if given and set(given) == set(speakers):
            return RoleAssignment(given, 1.0, {}, "given")
        if len(speakers) == 1:
            return RoleAssignment({speakers[0]: "customer"}, 0.0, {}, "single")

        hits: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for t in turns:
            hits[t.speaker] += len(self.agent.findall(t.text)) - 0.5 * len(self.customer.findall(t.text))
            counts[t.speaker] += 1
        scores = {s: hits[s] / counts[s] for s in speakers}
        scores[turns[0].speaker] += FIRST_SPEAKER_BONUS

        ranked = sorted(scores, key=scores.get, reverse=True)
        agent = ranked[0]
        top, second = scores[ranked[0]], scores[ranked[1]]
        confidence = min(1.0, max(0.0, (top - second) / (abs(top) + abs(second) + 1e-9)))
        roles = {s: ("agent" if s == agent else "customer") for s in speakers}
        return RoleAssignment(roles, round(confidence, 3), {k: round(v, 3) for k, v in scores.items()}, "rules")
