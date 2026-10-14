"""채널 통합 CX 리포트: CSAT·CES·NPS 추정 + 채널 비교 + 조용한 불만."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from importlib import resources

import pandas as pd
from jinja2 import Environment

from voclens.cx import compare
from voclens.cx.estimates import apply_estimates, save_estimates, session_features, train_csat
from voclens.text.taxonomy import normalize_domain

MODEL_NAME = "csat-ridge-v0.1"


def run_cx(conn: sqlite3.Connection, settings: dict, brand: str | None = None, domain: str | None = None):
    """특징 추출 → CSAT 학습·평가 → 추정치 저장.

    추정치는 전체 세션에 저장하고, brand·domain은 리포트 필터에만 쓴다.
    domain을 주면 그 도메인 세션끼리만 채널을 비교한다 (제품 영상의 배터리 불만을 배달 리뷰와 비교하지 않게).
    """
    features = session_features(conn, review_domain=settings["review"].get("domain"))
    if features.empty:
        raise ValueError("분석된 세션이 없습니다. 먼저 review / call / video 명령을 실행하세요.")
    model, evaluation = train_csat(features, seed=settings["report"]["seed"])
    estimates = apply_estimates(features, model)
    save_estimates(conn, estimates, MODEL_NAME if model else "none")
    mentions = compare.complaint_mentions(conn)
    domain = normalize_domain(domain)
    if domain:
        estimates = estimates[estimates["domain"] == domain]
        mentions = mentions[mentions["session_id"].isin(set(estimates["session_id"]))]
    if brand:
        estimates = estimates[estimates["brand"] == brand]
        mentions = mentions[mentions["brand"] == brand]
    return estimates, evaluation, mentions


def _mix_rows(mentions: pd.DataFrame, top_n: int = 12) -> tuple[list[str], list[dict]]:
    mix = compare.complaint_mix(mentions)
    if mix.empty:
        return [], []
    channels = [c for c in compare.CHANNEL_LABELS if c in mix.index]
    totals = mentions.groupby("label").size().sort_values(ascending=False)
    rows = []
    for label in totals.index[:top_n]:
        rows.append({"label": label, "n": int(totals[label]), **{c: float(mix.at[c, label]) for c in channels}})
    return channels, rows


def _outcome_rows(estimates: pd.DataFrame) -> list[dict]:
    calls = estimates[estimates["source_type"] == "call"]
    rows = []
    for outcome, g in calls.groupby("outcome"):
        rows.append({"outcome": outcome, "n": len(g), "ces": g["ces_est"].mean(), "csat": g["csat_est"].mean(), "frustration": g["frustration"].mean()})
    return sorted(rows, key=lambda r: r["ces"])


def build_cx_report(conn: sqlite3.Connection, settings: dict, brand: str | None = None, domain: str | None = "default") -> str:
    """domain: "default"면 설정의 리뷰 도메인, "all"이나 None이면 도메인 구분 없이 전체."""
    if domain == "default":
        domain = settings["review"].get("domain")
    elif domain == "all":
        domain = None
    estimates, evaluation, mentions = run_cx(conn, settings, brand, domain)
    positions = {
        name: compare.brand_positions(estimates, name) for name in ("happiness", "frustration")
    }
    brand_mix = estimates.groupby(["source_type", "brand"]).size().reset_index(name="n")
    channels, mix_rows = _mix_rows(mentions)

    context = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
        "brand": brand,
        "domain": normalize_domain(domain),
        "small_threshold": compare.SMALL_CHANNEL_MENTIONS,
        "labels": compare.CHANNEL_LABELS,
        "evaluation": evaluation,
        "summary": compare.channel_summary(estimates),
        "channels": channels,
        "mix_rows": mix_rows,
        "mix_test": compare.mix_test(mentions) if not mentions.empty else None,
        "gaps": compare.channel_gaps(mentions),
        "positions": {k: v for k, v in positions.items() if not v.empty},
        "brand_mix": brand_mix.to_dict("records"),
        "outcomes": _outcome_rows(estimates),
        "model_name": MODEL_NAME,
    }
    template = resources.files("voclens").joinpath("templates", "cx_report.md.j2").read_text(encoding="utf-8")
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["num"] = lambda x, d=1: "-" if x is None or pd.isna(x) else f"{x:.{d}f}"
    env.filters["signed"] = lambda x, d=2: "-" if x is None or pd.isna(x) else f"{x:+.{d}f}"
    return env.from_string(template).render(**context)
