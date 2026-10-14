"""개선 과제 리포트 (LLM 없이, 규칙 + 템플릿).

우선순위 점수 = 언급 비율(정규화) × (0.5 + 감정 강도) × (1 + 0.5 × 증가 추세) × (1 + 비즈니스 영향)
  - 감정 강도: 해당 불만 절의 max(스트레스, 좌절, 부정 정서) 평균 (0~1)
  - 증가 추세: 기간 마지막 1/3의 언급 비율이 앞 2/3보다 얼마나 늘었는지 (작성일이 있을 때만)
  - 비즈니스 영향: 해당 불만이 있는 리뷰의 이탈 의사 비율 초과분 + 재주문 의사 비율 부족분

4분면: 우선순위가 중앙값 이상이면 "영향 큼", 분류 기준의 난이도가 1이면 "실행 쉬움".
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources

import numpy as np
import pandas as pd
from jinja2 import Environment

from voicecx.cx import stats
from voicecx.schema import INDEX_LABELS
from voicecx.text.taxonomy import Taxonomy

REPORT_INDICES = ("stress", "happiness", "negative_affect", "frustration", "recovery", "engagement")
QUADRANTS = {
    (True, True): "Quick Win",
    (True, False): "핵심 과제",
    (False, True): "소규모 개선",
    (False, False): "모니터링",
}


@dataclass
class ReportData:
    sessions: pd.DataFrame
    units: pd.DataFrame
    labels: pd.DataFrame
    filters: dict = field(default_factory=dict)


def load_report_data(
    conn: sqlite3.Connection, brand: str | None = None, exclude_review_event: bool = False
) -> ReportData:
    sessions = pd.read_sql_query(
        """
        SELECT s.session_id, s.flags_json, s.peak_end, src.brand, src.store, src.rating, src.created_at
        FROM sessions s JOIN sources src ON src.source_id = s.source_id
        WHERE s.source_type = 'review'
        """,
        conn,
    )
    indices = pd.read_sql_query("SELECT session_id, index_name, raw FROM indices", conn)
    if not indices.empty:
        wide = indices.pivot(index="session_id", columns="index_name", values="raw").reset_index()
        sessions = sessions.merge(wide, on="session_id", how="left")
    sessions["review_event"] = sessions["flags_json"].fillna("").str.contains("review_event")

    units = pd.read_sql_query(
        "SELECT unit_id, session_id, ord, text, valence, aspect, aspect_sentiment, journey_stage,"
        " cx_signals_json, scores_json, confidence_json FROM units WHERE source_type = 'review'",
        conn,
    )
    scores = units["scores_json"].map(json.loads).apply(pd.Series)
    signals = units["cx_signals_json"].map(json.loads)
    units = pd.concat([units, scores], axis=1)
    for name in ("churn_intent", "not_recommend", "repurchase_intent", "recommend_intent"):
        units[name] = signals.map(lambda d, n=name: bool(d.get(n, False)))
    units["taxonomy_conf"] = units["confidence_json"].map(lambda s: json.loads(s).get("taxonomy", 0.0))

    labels = pd.read_sql_query(
        "SELECT unit_id, label, category FROM unit_labels WHERE label_type = 'complaint'", conn
    )

    filters = {}
    if brand:
        sessions = sessions[sessions["brand"] == brand]
        filters["브랜드"] = brand
    if exclude_review_event:
        sessions = sessions[~sessions["review_event"]]
        filters["리뷰 이벤트"] = "제외"
    units = units[units["session_id"].isin(sessions["session_id"])]
    labels = labels[labels["unit_id"].isin(units["unit_id"])]
    return ReportData(sessions.reset_index(drop=True), units.reset_index(drop=True), labels.reset_index(drop=True), filters)


def _session_flags(units: pd.DataFrame) -> pd.DataFrame:
    g = units.groupby("session_id")
    return pd.DataFrame(
        {
            "churn": g["churn_intent"].any() | g["not_recommend"].any(),
            "repurchase": g["repurchase_intent"].any(),
        }
    )


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def overview(data: ReportData, settings: dict) -> dict:
    s = data.sessions
    rep = settings["report"]
    rows = []
    for name in REPORT_INDICES:
        if name not in s:
            continue
        values = s[name].dropna()
        if values.empty:
            continue
        low, high = stats.bootstrap_ci(values, rep["bootstrap_iterations"], seed=rep["seed"])
        rows.append({"name": INDEX_LABELS[name], "n": len(values), "mean": values.mean(), "low": low, "high": high})
    return {
        "n_reviews": len(s),
        "n_units": len(data.units),
        "review_event_share": float(s["review_event"].mean()) if len(s) else 0.0,
        "mean_rating": float(s["rating"].mean()) if s["rating"].notna().any() else None,
        "indices": rows,
        "peak_end": float(s["peak_end"].mean()) if s["peak_end"].notna().any() else None,
    }


def brand_comparison(data: ReportData, settings: dict) -> dict | None:
    s = data.sessions.dropna(subset=["brand"])
    brands = s["brand"].value_counts()
    if len(brands) < 2:
        return None
    rep = settings["report"]
    rows = []
    for brand, n in brands.items():
        group = s[s["brand"] == brand]
        row = {"brand": brand, "n": int(n), "small": n < rep["min_n"]}
        for name in ("stress", "happiness", "frustration"):
            global_mean = s[name].mean()
            row[name] = stats.shrink_mean(group[name].mean(), int(group[name].notna().sum()), global_mean, rep["shrinkage_k"])
        rows.append(row)

    # 불만 유형 구성비 차이 (리뷰 단위 언급 수)
    merged = data.labels.merge(data.units[["unit_id", "session_id"]], on="unit_id").merge(
        s[["session_id", "brand"]], on="session_id"
    )
    complaints = merged[merged["category"].notna()]
    tests = {}
    counts = complaints.drop_duplicates(["session_id", "label"]).pivot_table(
        index="brand", columns="label", values="session_id", aggfunc="count", fill_value=0
    )
    if counts.shape[0] >= 2 and counts.shape[1] >= 2:
        tests["complaint_mix"] = stats.compare_proportions(counts.values)

    top_two = list(brands.index[:2])
    for name in ("stress", "happiness"):
        a = s.loc[s["brand"] == top_two[0], name].dropna()
        b = s.loc[s["brand"] == top_two[1], name].dropna()
        if len(a) >= 3 and len(b) >= 3:
            tests[name] = stats.compare_distributions(a, b)
    return {"rows": rows, "tests": tests, "top_two": top_two}


def pain_points(data: ReportData, taxonomy: Taxonomy, top_n: int = 8) -> list[dict]:
    units, labels, sessions = data.units, data.labels, data.sessions
    if units.empty or labels.empty:
        return []
    complaints = labels[labels["label"].map(taxonomy.is_complaint)].merge(units, on="unit_id")
    if complaints.empty:
        return []

    total = len(sessions)
    flags = _session_flags(units)
    churn_all = float(flags["churn"].mean())
    rep_all = float(flags["repurchase"].mean())

    dates = _parse_dates(sessions.set_index("session_id")["created_at"])
    use_trend = dates.notna().mean() >= 0.7 and dates.nunique() >= 3
    cutoff = dates.quantile(2 / 3) if use_trend else None

    items = []
    for label, group in complaints.groupby("label"):
        session_ids = set(group["session_id"])
        share = len(session_ids) / total
        intensity = float(group[["stress", "frustration", "negative_affect"]].max(axis=1).mean())
        f = flags.loc[list(session_ids)]
        churn_with, rep_with = float(f["churn"].mean()), float(f["repurchase"].mean())
        impact = max(0.0, churn_with - churn_all) + max(0.0, rep_all - rep_with)

        trend = None
        if use_trend:
            early_ids = set(dates[dates < cutoff].index)
            late_ids = set(dates[dates >= cutoff].index)
            if early_ids and late_ids:
                early = len(session_ids & early_ids) / len(early_ids)
                late = len(session_ids & late_ids) / len(late_ids)
                trend = float(np.clip((late - early) / max(early, 0.05), -1, 2))

        median_len = group["text"].str.len().median()
        ranked = group.assign(_dist=(group["text"].str.len() - median_len).abs()).sort_values(
            ["taxonomy_conf", "_dist"], ascending=[False, True]
        )
        representative = list(dict.fromkeys(ranked["text"]))[:3]
        worst = group.loc[(group["stress"] + group["frustration"]).idxmax(), "text"]

        items.append(
            {
                "label": label,
                "category": taxonomy.category_of(label),
                "n_reviews": len(session_ids),
                "share": share,
                "intensity": intensity,
                "churn_with": churn_with,
                "churn_all": churn_all,
                "rep_with": rep_with,
                "rep_all": rep_all,
                "impact": impact,
                "trend": trend,
                "difficulty": taxonomy.difficulty_of(label),
                "evidence": representative,
                "worst": worst,
            }
        )

    max_share = max(i["share"] for i in items)
    for i in items:
        trend_boost = 1 + 0.5 * max(i["trend"] or 0.0, 0.0)
        i["_raw"] = (i["share"] / max_share) * (0.5 + i["intensity"]) * trend_boost * (1 + i["impact"])
    max_raw = max(i["_raw"] for i in items)
    for i in items:
        i["priority"] = 100 * i["_raw"] / max_raw if max_raw else 0.0

    median_priority = float(np.median([i["priority"] for i in items]))
    for i in items:
        high = i["priority"] >= median_priority
        easy = i["difficulty"] <= 1
        i["quadrant"] = QUADRANTS[(high, easy)]

    return sorted(items, key=lambda i: -i["priority"])[:top_n]


def journey_table(data: ReportData, min_n: int) -> list[dict]:
    u = data.units.dropna(subset=["journey_stage"])
    rows = []
    for stage, g in u.groupby("journey_stage"):
        rows.append(
            {
                "stage": stage,
                "n": len(g),
                "valence": g["valence"].mean(),
                "stress": g["stress"].mean() * 100,
                "frustration": g["frustration"].mean() * 100,
                "happiness": g["happiness"].mean() * 100,
                "small": len(g) < min_n,
            }
        )
    return sorted(rows, key=lambda r: r["valence"])


def aspect_table(data: ReportData) -> list[dict]:
    u = data.units.dropna(subset=["aspect"])
    rows = []
    for aspect, g in u.groupby("aspect"):
        rows.append(
            {
                "aspect": aspect,
                "n": len(g),
                "sentiment": g["aspect_sentiment"].mean(),
                "positive": int((g["aspect_sentiment"] > 0.1).sum()),
                "negative": int((g["aspect_sentiment"] < -0.1).sum()),
            }
        )
    return sorted(rows, key=lambda r: r["sentiment"])


def mismatches(data: ReportData, top_n: int = 5, z_threshold: float = 1.5) -> list[dict]:
    """별점과 감정 방향이 반대인 리뷰. 잔차가 크면서 방향도 어긋나야 고른다.

    잔차만 보면 "1점 + 약한 불만" 같은 평범한 리뷰도 걸리기 때문에,
    (별점이 높게 튀었는데 감정은 부정) 또는 (별점이 낮게 튀었는데 감정은 긍정)인 경우만 남긴다.
    """
    valence = data.units.groupby("session_id")["valence"].mean().rename("mean_valence")
    s = data.sessions.join(valence, on="session_id").dropna(subset=["rating", "mean_valence"])
    if len(s) < 10:
        return []
    s = s.assign(z=stats.rating_mismatch(s["rating"], s["mean_valence"]))
    conflict = ((s["z"] >= z_threshold) & (s["mean_valence"] < -0.05)) | (
        (s["z"] <= -z_threshold) & (s["mean_valence"] > 0.05)
    )
    picked = s[conflict].reindex(s.loc[conflict, "z"].abs().sort_values(ascending=False).index).head(top_n)
    texts = data.units.sort_values("ord").groupby("session_id")["text"].apply(" / ".join)
    return [
        {
            "rating": r.rating,
            "valence": r.mean_valence,
            "z": r.z,
            "review_event": r.review_event,
            "text": texts.get(r.session_id, ""),
        }
        for r in picked.itertuples()
    ]


def review_event_bias(data: ReportData) -> dict | None:
    s = data.sessions
    if s["review_event"].sum() < 3 or (~s["review_event"]).sum() < 3:
        return None
    event = s[s["review_event"]]
    normal = s[~s["review_event"]]
    return {
        "n_event": len(event),
        "n_normal": len(normal),
        "rating_event": event["rating"].mean(),
        "rating_normal": normal["rating"].mean(),
        "happiness_event": event["happiness"].mean(),
        "happiness_normal": normal["happiness"].mean(),
        "test": stats.compare_distributions(event["happiness"].dropna(), normal["happiness"].dropna()),
    }


def build_report(conn: sqlite3.Connection, settings: dict, brand: str | None = None, exclude_review_event: bool = False) -> str:
    taxonomy = Taxonomy.load()
    data = load_report_data(conn, brand=brand, exclude_review_event=exclude_review_event)
    if data.sessions.empty:
        raise ValueError("리포트를 만들 리뷰가 없습니다. 먼저 `voicecx review <csv>`를 실행하세요.")
    min_n = settings["report"]["min_n"]
    models = conn.execute("SELECT models_json FROM runs WHERE status = 'done' ORDER BY started_at DESC LIMIT 1").fetchone()

    context = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
        "filters": data.filters,
        "overview": overview(data, settings),
        "brands": brand_comparison(data, settings),
        "pains": pain_points(data, taxonomy),
        "journey": journey_table(data, min_n),
        "aspects": aspect_table(data),
        "mismatches": mismatches(data),
        "event_bias": review_event_bias(data),
        "min_n": min_n,
        "models": json.loads(models["models_json"]) if models else {},
    }
    template = resources.files("voicecx").joinpath("templates", "review_report.md.j2").read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["pct"] = lambda x: "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"
    env.filters["num"] = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x:.{d}f}"
    return env.from_string(template).render(**context)
