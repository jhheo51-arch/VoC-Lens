"""리뷰·전사 텍스트 정제.

- 반복 문자 정규화: "ㅋㅋㅋㅋ" → "ㅋㅋ"
- 이모티콘·이모지를 감정 토큰으로: "ㅠㅠ" → [슬픔]
- 개인정보 마스킹: 전화번호, 이메일, 긴 숫자
- 리뷰 이벤트 리뷰 표시: 긍정 편향의 주요 원인이라 집계에서 따로 본다
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 숫자는 건드리지 않는다 ("10000원"이 "100원"이 되면 안 됨)
_REPEAT_CHAR = re.compile(r"([^\d\s])\1{2,}")
_REPEAT_WORD = re.compile(r"([^\d\s]{2,4}?)\1{2,}")

# 자모 이모티콘은 반복 정규화 이후 2글자 이상으로 남는다
_JAMO_CUES = [
    (re.compile(r"[ㅠㅜ]{2,}"), "[슬픔]"),
    (re.compile(r"[ㅋ]{2,}"), "[웃음]"),
    (re.compile(r"[ㅎ]{2,}"), "[웃음]"),
]
_EMOJI_CUES = {
    "[기쁨]": "😀😁😂🤣😃😄😆😊😋😍🥰😘☺🙂🤗👍👏💕❤♥💯✨🎉",
    "[슬픔]": "😢😭😞😔😟🙁☹😥😿💔",
    "[분노]": "😠😡🤬👎💢😤",
    "[혐오]": "🤢🤮",
    "[놀람]": "😮😲😱😳",
}
_EMOJI_TO_TOKEN = {ch: token for token, chars in _EMOJI_CUES.items() for ch in chars}

_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[이메일]"),
    (re.compile(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}"), "[전화번호]"),
    (re.compile(r"0\d{1,2}-\d{3,4}-\d{4}"), "[전화번호]"),
    (re.compile(r"\d{8,}"), "[번호]"),
]

_REVIEW_EVENT = re.compile(
    r"리뷰\s*이벤트|리뷰\s*약속|리뷰\s*참여|리뷰\s*서비스"
    r"|서비스\s*(감사|잘\s*받|주셔서)"
    r"|(콜라|사이다|음료|감자튀김|치즈볼|만두)\s*(서비스|증정)"
)


@dataclass
class Preprocessed:
    text: str
    flags: list[str] = field(default_factory=list)


def normalize_repeats(text: str) -> str:
    text = _REPEAT_CHAR.sub(r"\1\1", text)
    return _REPEAT_WORD.sub(r"\1\1", text)


def replace_emotion_cues(text: str) -> str:
    for pattern, token in _JAMO_CUES:
        text = pattern.sub(f" {token} ", text)
    out = []
    for ch in text:
        token = _EMOJI_TO_TOKEN.get(ch)
        out.append(f" {token} " if token else ch)
    return "".join(out)


def mask_pii(text: str) -> str:
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def detect_review_event(text: str) -> bool:
    return bool(_REVIEW_EVENT.search(text))


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> Preprocessed:
    flags: list[str] = []
    if detect_review_event(text):
        flags.append("review_event")

    masked = mask_pii(text)
    if masked != text:
        flags.append("pii_masked")

    cleaned = replace_emotion_cues(normalize_repeats(masked))
    return Preprocessed(text=_collapse_spaces(cleaned), flags=flags)
