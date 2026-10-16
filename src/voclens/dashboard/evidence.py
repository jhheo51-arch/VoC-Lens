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
    # 같은 세션의 앞 문장과 이어 볼 때만 걸리는 불만 표현("블루를 샀는데" + "블랙이 왔어요")도 근거로 보여 준다
    sessions = df["session_id"].tolist() if "session_id" in df else [None] * len(df)
    prevs = [texts.iloc[i - 1] if i and sessions[i] is not None and sessions[i] == sessions[i - 1] else "" for i in range(len(df))]
    df["불만 근거"] = [taxonomy.explain(t, v, domain, p) for t, v, p in zip(texts, valences, prevs)]
    df["신호 근거"] = texts.map(cx.explain)
    return df
