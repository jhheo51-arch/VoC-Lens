"""2주차 라벨링 도구 (docs/analysis_design.md 1.3, labeling/guidelines.md).

  prepare    리뷰 CSV → 절 분리·현재 분류 → 모델이 헷갈리는 절부터 N개 골라 Label Studio 작업 파일(JSON)
  import     Label Studio 내보내기(JSON) → 절 ID별 라벨 CSV (기본은 원문 없이 ID + 라벨만)
  agreement  같은 절을 두 번 라벨링한 결과로 자기 일치도(Cohen's κ) 계산

헷갈리는 정도(margin, 작을수록 먼저 라벨링)
  - 임베딩 인코더가 있으면: 시드 문장 중심과의 코사인 유사도 1위 − 2위 (EmbeddingTaxonomyClassifier)
  - 없으면 규칙 기반 대리 지표: 여러 유형이 동시에 붙음 < 분류 불가 < 주제 표현만으로 붙음
    < 부정 감성인데 유형 없음 < 감정 없는 중립 < 칭찬 < 불만 표현 하나로 확실
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from voclens.ingest.review_csv import load_reviews
from voclens.pipelines.review import ReviewModels, analyze_records
from voclens.text.taxonomy import EmbeddingTaxonomyClassifier

PROXY_MARGIN = {
    "multi": 0.10,
    "unknown": 0.15,
    "topic_only": 0.20,
    "negative_unlabeled": 0.25,
    "neutral": 0.60,
    "praise": 0.70,
    "certain": 0.90,
}


def clause_table(csv_path: str | Path, settings: dict, domain: str | None = None) -> pd.DataFrame:
    """리뷰 CSV를 분석해 절 단위 표로 만든다 (현재 규칙 기반 분류 포함)."""
    local = {**settings, "review": {**settings["review"], "domain": domain or settings["review"].get("domain")}}
    records = load_reviews(csv_path, local["review"]["columns"])
    sessions = analyze_records(records, local, ReviewModels.default())
    rows = []
    for s in sessions:
        for u in s.units:
            rows.append(
                {
                    "unit_id": u.unit_id,
                    "review_id": s.source_id,
                    "text": u.text,
                    "review": s.text,
                    "valence": u.valence,
                    "labels": u.complaint_categories,
                    "taxonomy_conf": u.confidence.get("taxonomy", 0.0),
                }
            )
    return pd.DataFrame(rows)


def _proxy_kind(labels: list[str], conf: float, valence: float) -> str:
    complaints = [l for l in labels if l not in ("칭찬", "분류 불가")]
    if len(complaints) >= 2:
        return "multi"
    if "분류 불가" in labels:
        return "unknown"
    if complaints and conf < 1.0:
        return "topic_only"
    if complaints:
        return "certain"
    if "칭찬" in labels:
        return "praise"
    if valence < 0:
        return "negative_unlabeled"
    return "neutral"


def score_uncertainty(df: pd.DataFrame, encode=None) -> pd.DataFrame:
    """margin(작을수록 헷갈림)과 zero_shot(현재 예측) 열을 붙인다."""
    out = df.copy()
    if encode is not None:
        clf = EmbeddingTaxonomyClassifier(encode=encode, threshold=0.0)
        results = clf.classify(out["text"].tolist())
        out["zero_shot"] = [label or "" for label, _, _ in results]
        out["margin"] = [round(margin, 4) for _, _, margin in results]
        out["basis"] = "embedding"
    else:
        kinds = [_proxy_kind(l, c, v or 0.0) for l, c, v in zip(out["labels"], out["taxonomy_conf"], out["valence"])]
        out["kind"] = kinds
        out["margin"] = [PROXY_MARGIN[k] for k in kinds]
        out["zero_shot"] = [", ".join(l) for l in out["labels"]]
        out["basis"] = "rules"
    return out


def select_for_labeling(df: pd.DataFrame, n: int = 500, per_review: int = 2, seed: int = 42) -> pd.DataFrame:
    """헷갈리는 절부터 n개. 같은 문장(공백 무시)은 한 번만, 한 리뷰에서 per_review개까지만 골라 다양성을 지킨다.
    같은 margin 안에서는 무작위."""
    rng = np.random.default_rng(seed)
    shuffled = df.assign(_tie=rng.random(len(df)), _norm=df["text"].str.replace(r"\s+", "", regex=True)).sort_values(["margin", "_tie"])
    unique = shuffled.drop_duplicates("_norm")
    picked = unique.groupby("review_id", sort=False).head(per_review)
    return picked.head(n).drop(columns=["_tie", "_norm"]).reset_index(drop=True)


def to_label_studio(df: pd.DataFrame) -> list[dict]:
    """labeling/label_studio_config.xml의 변수($text, $review, $zero_shot, $margin)에 맞춘 작업 목록."""
    return [
        {
            "data": {
                "unit_id": r.unit_id,
                "text": r.text,
                "review": r.review,
                "zero_shot": r.zero_shot or "(없음)",
                "margin": float(r.margin),
            }
        }
        for r in df.itertuples()
    ]


def read_label_studio_export(path: str | Path) -> pd.DataFrame:
    """Label Studio JSON 내보내기 → unit_id, labels(목록), hold, memo, text."""
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations") or []
        if not annotations:
            continue
        result = annotations[-1].get("result", [])   # 마지막(최종) 주석
        labels, hold, memo = [], False, ""
        for item in result:
            value = item.get("value", {})
            if item.get("from_name") == "complaint":
                labels = list(value.get("choices", []))
            elif item.get("from_name") == "hold":
                hold = "보류" in value.get("choices", [])
            elif item.get("from_name") == "memo":
                memo = " ".join(value.get("text", []))
        rows.append({"unit_id": data.get("unit_id"), "labels": labels, "hold": hold, "memo": memo, "text": data.get("text", "")})
    return pd.DataFrame(rows, columns=["unit_id", "labels", "hold", "memo", "text"])


def labels_to_csv(df: pd.DataFrame, out: str | Path, with_text: bool = False) -> Path:
    """라벨 CSV 저장. 외부 데이터 재배포를 막으려고 기본은 원문을 빼고 ID와 라벨만 쓴다."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = df.assign(labels=df["labels"].map(lambda l: "|".join(l)))
    cols = ["unit_id", "labels", "hold", "memo"] + (["text"] if with_text else [])
    table[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return out


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def agreement(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    """두 라벨링 결과의 자기 일치도. 다중 라벨이라 유형마다 '붙였나/안 붙였나'의 κ를 내고 평균한다.

    보류한 절은 뺀다. 목표는 macro κ 0.7 이상 (docs/analysis_design.md 1.3).
    """
    a = first[~first["hold"]].set_index("unit_id")["labels"]
    b = second[~second["hold"]].set_index("unit_id")["labels"]
    common = a.index.intersection(b.index)
    labels = sorted({l for ls in a.loc[common] for l in ls} | {l for ls in b.loc[common] for l in ls})
    per_label = {}
    for label in labels:
        k = cohen_kappa([label in a[i] for i in common], [label in b[i] for i in common])
        per_label[label] = None if k is None else round(k, 3)
    valid = [k for k in per_label.values() if k is not None]
    exact = float(np.mean([set(a[i]) == set(b[i]) for i in common])) if len(common) else 0.0
    return {
        "n_common": int(len(common)),
        "exact_match": round(exact, 3),
        "macro_kappa": round(float(np.mean(valid)), 3) if valid else None,
        "per_label": per_label,
    }
