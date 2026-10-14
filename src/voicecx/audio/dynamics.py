"""대화 역학 지표 (좌절·회복 지수의 핵심 재료).

발화 점유율, 끼어들기, 침묵·대기, 같은 말 반복, 상담원 공감·해결 표현, 통화 결과.
전사본의 타임스탬프와 텍스트만 있으면 계산된다 (음성 파일 없이도 가능).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voicecx.config import load_yaml


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def char_bigrams(text: str) -> set[str]:
    s = re.sub(r"\s+", "", text)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    """|A∩B| / min(|A|,|B|). 같은 말을 다른 말에 섞어 반복해도(길이가 달라도) 잡히도록 Jaccard 대신 쓴다."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class TurnFlags:
    interruption: bool = False
    repeat_of: int | None = None      # 비슷한 이전 고객 발화의 인덱스
    empathy: bool = False
    resolution: bool = False
    transfer: bool = False
    pending: bool = False
    silence_before: float = 0.0


@dataclass
class DynamicsResult:
    duration: float
    customer_talk_ratio: float
    customer_interruptions: int
    agent_interruptions: int
    total_silence: float
    max_silence: float
    n_long_silence: int
    repeat_count: int
    empathy_count: int
    outcome: str                      # resolved | transferred | pending | unknown
    turn_flags: list[TurnFlags] = field(default_factory=list)

    def as_meta(self) -> dict:
        return {
            "duration": round(self.duration, 2),
            "customer_talk_ratio": round(self.customer_talk_ratio, 3),
            "customer_interruptions": self.customer_interruptions,
            "agent_interruptions": self.agent_interruptions,
            "total_silence": round(self.total_silence, 2),
            "max_silence": round(self.max_silence, 2),
            "n_long_silence": self.n_long_silence,
            "repeat_count": self.repeat_count,
            "empathy_count": self.empathy_count,
            "outcome": self.outcome,
        }


class DynamicsAnalyzer:
    name = "dynamics-v0.1"

    def __init__(self, settings: dict, phrases: dict | None = None):
        data = phrases or load_yaml("lexicons/call_phrases.yaml")
        self.empathy = _compile(data["empathy"])
        self.resolution = _compile(data["resolution"])
        self.transfer = _compile(data["transfer"])
        self.pending = _compile(data["pending"])
        c = settings["call"]
        self.long_silence = c["long_silence_sec"]
        self.overlap = c["interruption_overlap_sec"]
        self.repeat_sim = c["repeat_similarity"]
        self.repeat_min_chars = c["repeat_min_chars"]

    def _outcome(self, turns, roles, flags) -> str:
        agent_idx = [i for i, t in enumerate(turns) if roles.get(t.speaker) == "agent"]
        if not agent_idx:
            return "unknown"
        # 통화 후반부(상담원 발화의 뒤쪽 40%)의 마지막 조치로 결과를 정한다
        tail = agent_idx[int(len(agent_idx) * 0.6) :] or agent_idx
        for i in reversed(tail):
            f = flags[i]
            if f.transfer:
                return "transferred"
            if f.pending:
                return "pending"
            if f.resolution:
                return "resolved"
        return "unknown"

    def analyze(self, turns, roles: dict[str, str]) -> DynamicsResult:
        flags = [TurnFlags() for _ in turns]
        talk = {"customer": 0.0, "agent": 0.0}
        interruptions = {"customer": 0, "agent": 0}
        silences: list[float] = []
        customer_history: list[tuple[int, set[str]]] = []
        last_end = None
        prev = None

        for i, t in enumerate(turns):
            role = roles.get(t.speaker, "customer")
            talk[role] = talk.get(role, 0.0) + (t.end - t.start)
            f = flags[i]

            if last_end is not None:
                gap = t.start - last_end
                if gap >= self.long_silence:
                    silences.append(gap)
                    f.silence_before = round(gap, 2)
            if prev is not None and prev.speaker != t.speaker and t.start < prev.end - self.overlap:
                f.interruption = True
                interruptions[role] = interruptions.get(role, 0) + 1

            if role == "customer":
                grams = char_bigrams(t.text)
                if len(re.sub(r"\s+", "", t.text)) >= self.repeat_min_chars:
                    for j, past in customer_history:
                        if overlap_coefficient(grams, past) >= self.repeat_sim:
                            f.repeat_of = j
                            break
                    customer_history.append((i, grams))
            else:
                f.empathy = bool(self.empathy.search(t.text))
                f.resolution = bool(self.resolution.search(t.text))
                f.transfer = bool(self.transfer.search(t.text))
                f.pending = bool(self.pending.search(t.text))

            last_end = t.end if last_end is None else max(last_end, t.end)
            prev = t

        total_talk = talk["customer"] + talk["agent"]
        return DynamicsResult(
            duration=max(t.end for t in turns) - min(t.start for t in turns),
            customer_talk_ratio=talk["customer"] / total_talk if total_talk else 0.0,
            customer_interruptions=interruptions["customer"],
            agent_interruptions=interruptions["agent"],
            total_silence=sum(silences),
            max_silence=max(silences, default=0.0),
            n_long_silence=len(silences),
            repeat_count=sum(f.repeat_of is not None for f in flags),
            empathy_count=sum(f.empathy for f in flags),
            outcome=self._outcome(turns, roles, flags),
            turn_flags=flags,
        )
