"""문장·절 분리 (규칙 기반 기준선).

리뷰는 문장부호 없이 "맛있어요 배달도 빨라요"처럼 쓰는 경우가 많아서,
어절 끝이 종결어미(요/다/음/임/함/됨/네/죠)인지로 문장을 나눈다.
한 문장 안의 상반된 평가("맛은 좋은데 배달이 늦어요")는 대조 표현에서 절로 나눈다.

감정 토큰([웃음], [슬픔] 등)은 앞 문장에 붙인다.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"^\[[가-힣]+\]$")
_PUNCT_END = re.compile(r"[.!?…~]+$")
_EOMI_END = re.compile(r"[가-힣][요다음임함됨네죠][,]?$")
# 종결어미처럼 보이지만 명사·조사인 어절
_NOT_SENTENCE_END = ("보다", "처음", "마음", "다음", "웃음", "얼음", "걸음", "믿음", "필요", "중요", "주요", "동네", "포함", "함께")

_CLAUSE_END = re.compile(r"(데|지만)[,]?$")
_CLAUSE_END_EXCEPT = {"근데", "그런데", "군데"}
_CLAUSE_START = {"근데", "그런데", "하지만", "그러나", "반면", "반면에", "그래도"}

_HAS_CONTENT = re.compile(r"[가-힣A-Za-z0-9]")


def _is_token(word: str | None) -> bool:
    return bool(word) and bool(_TOKEN.match(word))


def _ends_sentence(word: str) -> bool:
    if _PUNCT_END.search(word):
        return True
    bare = word.rstrip(",")
    if bare.endswith(_NOT_SENTENCE_END):
        return False
    return bool(_EOMI_END.search(word))


def split_sentences(text: str) -> list[str]:
    words = text.split()
    sentences: list[list[str]] = []
    current: list[str] = []
    pending_end = False

    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _is_token(word):
            ends = pending_end
        else:
            ends = _ends_sentence(word)
        if ends and _is_token(nxt):
            pending_end = True
            continue
        if ends:
            sentences.append(current)
            current = []
        pending_end = False

    if current:
        sentences.append(current)
    return [" ".join(s) for s in sentences]


def split_clauses(sentence: str) -> list[str]:
    words = sentence.split()
    clauses: list[list[str]] = []
    current: list[str] = []

    for word in words:
        if word in _CLAUSE_START and current:
            clauses.append(current)
            current = []
        current.append(word)
        bare = word.rstrip(",")
        if bare not in _CLAUSE_END_EXCEPT and len(bare) >= 2 and _CLAUSE_END.search(word):
            clauses.append(current)
            current = []

    if current:
        clauses.append(current)
    return [" ".join(c) for c in clauses]


def _has_content(clause: str) -> bool:
    without_tokens = re.sub(r"\[[가-힣]+\]", "", clause)
    return bool(_HAS_CONTENT.search(without_tokens))


def segment(text: str) -> list[str]:
    """정제된 텍스트를 절 목록으로 나눈다. 내용이 없는 절은 버린다."""
    clauses = [c for s in split_sentences(text) for c in split_clauses(s)]
    return [c for c in clauses if _has_content(c)]
