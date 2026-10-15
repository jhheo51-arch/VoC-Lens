"""문장별 판단 근거: 어떤 표현 때문에 그 감정·불만 유형·CX 신호로 판단했는지.

분석 때와 같은 규칙(사전·분류 기준)으로 다시 계산해서 보여준다. 근거를 DB에 따로 저장하지 않으므로
규칙을 고치면 근거도 자동으로 바뀐다.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from voclens.text.cx_signals import CxSignalDetector
from voclens.text.emotion import LexiconEmotionModel
from voclens.text.taxonomy import KeywordTaxonomyClassifier


@lru_cache(maxsize=1)
def _models():
    return LexiconEmotionModel(), KeywordTaxonomyClassifier(), CxSignalDetector()


def add_evidence(units: pd.DataFrame, domain: str | None = None, text_col: str = "text", valence_col: str = "valence") -> pd.DataFrame:
    """units에 감정 근거·불만 근거·신호 근거 열을 붙인다."""
    emotion, taxonomy, cx = _models()
    df = units.copy()
    texts = df[text_col].fillna("").astype(str)
    valences = df[valence_col].fillna(0.0).astype(float) if valence_col in df else pd.Series(0.0, index=df.index)
    df["감정 근거"] = texts.map(emotion.explain)
    df["불만 근거"] = [taxonomy.explain(t, v, domain) for t, v in zip(texts, valences)]
    df["신호 근거"] = texts.map(cx.explain)
    return df
