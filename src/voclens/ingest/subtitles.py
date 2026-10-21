"""자막(VTT/SRT) 파싱과 문장 단위 정렬.

자막 큐(cue)는 문장 중간에서 끊기는 경우가 많다. 그래서 큐 텍스트를 이어 붙여 문장을 나눈 뒤,
글자 위치로 각 문장의 시작·끝 시각을 되찾는다.
YouTube 자막의 "앞 줄 반복"(롤링 자막)도 제거한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voclens.text.segment import split_sentences

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    m = _TIME.search(value)
    if not m:
        raise ValueError(f"시각 형식이 아닙니다: {value}")
    h, mnt, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _parse_blocks(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        timing = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timing is None:
            continue
        left, right = lines[timing].split("-->", 1)
        body = " ".join(_TAG.sub("", l).strip() for l in lines[timing + 1 :])
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            cues.append(Cue(parse_timestamp(left), parse_timestamp(right.strip().split(" ")[0]), body))
    return cues


def dedupe_rolling(cues: list[Cue]) -> list[Cue]:
    """앞 큐 내용이 다음 큐 앞부분에 반복되면 반복된 부분을 지운다."""
    out: list[Cue] = []
    prev_raw = None   # 반복 제거 전의 이전 큐 원문과 비교해야 한다
    for cue in cues:
        text = cue.text
        if prev_raw is not None and out:
            if text == prev_raw:
                out[-1].end = cue.end
                continue
            if text.startswith(prev_raw):
                text = text[len(prev_raw) :].strip()
        prev_raw = cue.text
        if text:
            out.append(Cue(cue.start, cue.end, text))
    return out


def parse_subtitles(path_or_text: str | Path) -> list[Cue]:
    """VTT 또는 SRT 파일 경로(또는 내용 문자열)를 큐 목록으로."""
    text = str(path_or_text)
    if "\n" not in text and Path(text).exists():
        text = Path(text).read_text(encoding="utf-8-sig")
    return dedupe_rolling(_parse_blocks(text))


@dataclass
class TimedSentence:
    start: float
    end: float
    text: str


def cues_to_sentences(cues: list[Cue], max_gap: float = 5.0) -> list[TimedSentence]:
    """큐를 이어 붙여 문장으로 나누고, 문장마다 시작·끝 시각을 붙인다.

    큐 사이가 max_gap초보다 벌어지면 거기서 끊는다. 자막에 마침표가 없으면
    몇 분 떨어진 큐가 한 문장으로 붙기 때문이다.
    """
    out: list[TimedSentence] = []
    group: list[Cue] = []
    for cue in cues:
        if group and cue.start - group[-1].end > max_gap:
            out.extend(_group_to_sentences(group))
            group = []
        group.append(cue)
    if group:
        out.extend(_group_to_sentences(group))
    return out


def _group_to_sentences(cues: list[Cue]) -> list[TimedSentence]:
    spans = []           # (글자 시작, 글자 끝, 큐)
    parts = []
    pos = 0
    for cue in cues:
        words = cue.text.split()
        chunk = " ".join(words)
        spans.append((pos, pos + len(chunk), cue))
        parts.append(chunk)
        pos += len(chunk) + 1
    full = " ".join(parts)

    def cue_at(char: int) -> Cue:
        for s, e, cue in spans:
            if char <= e:
                return cue
        return spans[-1][2]

    out = []
    cursor = 0
    for sentence in split_sentences(full):
        start = full.find(sentence, cursor)
        end = start + len(sentence) - 1
        cursor = end + 1
        out.append(TimedSentence(cue_at(start).start, cue_at(end).end, sentence))
    return out
