"""채널 비교 (docs/analysis_design.md 6.3).

원점수를 그대로 비교하지 않는다. 리뷰는 리뷰 이벤트 때문에 긍정 쪽으로, 상담은 문제가 있을 때
걸기 때문에 부정 쪽으로 치우쳐 있다. 그래서 다음 두 가지로 비교한다.
  1. 불만 유형 구성비: 채널 안에서 불만 언급 중 각 유형이 차지하는 비율 (합 100%)
  2. 채널 내 표준화: 브랜드 평균 지수를 그 채널 분포 안의 z-score로 바꿔 채널을 넘어 위치를 비교

"조용한 불만": 상담·영상 댓글에서는 비중이 큰데 리뷰에서는 작은 유형.
  리뷰만 보는 조직은 놓치기 쉬운 불만이다. (반대 방향은 "리뷰에서만 큰 불만"으로 따로 표시)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from voclens.cx import stats

CHANNEL_LABELS = {"review": "리뷰", "call": "상담", "video": "영상(크리에이터)", "comment": "영상 댓글"}


def complaint_mentions(conn: sqlite3.Connection) -> pd.DataFrame:
    """불만 언급(발화 × 불만 유형) 목록. 영상 댓글은 '영상 댓글' 채널로 분리한다."""
    df = pd.read_sql_query(
        """
        SELECT u.unit_id, u.session_id, u.source_type, u.speaker_role, l.label, l.category, src.brand
        FROM unit_labels l
        JOIN units u ON u.unit_id = l.unit_id
        JOIN sessions s ON s.session_id = u.session_id
        JOIN sources src ON src.source_id = s.source_id
        WHERE l.label_type = 'complaint' AND l.category IS NOT NULL
          AND NOT (u.source_type = 'call' AND u.speaker_role = 'agent')
        """,
        conn,
    )
    df["channel"] = np.where(
        (df["source_type"] == "video") & (df["speaker_role"] == "commenter"), "comment", df["source_type"]
    )
    return df


def complaint_mix(mentions: pd.DataFrame) -> pd.DataFrame:
    """채널 × 불만 유형 구성비(%). 행 합이 100."""
    if mentions.empty:
        return pd.DataFrame()
    counts = mentions.pivot_table(index="channel", columns="label", values="unit_id", aggfunc="count", fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def mix_test(mentions: pd.DataFrame):
    counts = mentions.pivot_table(index="channel", columns="label", values="unit_id", aggfunc="count", fill_value=0)
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        return None
    return stats.compare_proportions(counts.values)


# 채널의 전체 불만 언급이 이보다 적으면 "표본 적음"으로 표시한다
SMALL_CHANNEL_MENTIONS = 20


@dataclass
class ChannelGap:
    label: str
    channel: str
    channel_share: float
    review_share: float
    ratio: float | None
    n_mentions: int
    channel_total: int = 0

    @property
    def small_sample(self) -> bool:
        return self.channel_total < SMALL_CHANNEL_MENTIONS


def channel_gaps(mentions: pd.DataFrame, ratio: float = 2.0, min_share: float = 10.0, min_mentions: int = 2) -> dict[str, list[ChannelGap]]:
    """조용한 불만(다른 채널 ≫ 리뷰, 채널마다)과 리뷰에서만 큰 불만(리뷰 ≫ 다른 채널 합계)."""
    mix = complaint_mix(mentions)
    out = {"silent": [], "review_only": []}
    if mix.empty or "review" not in mix.index:
        return out
    counts = mentions.groupby(["channel", "label"]).size()
    totals = mentions.groupby("channel").size()
    review = mix.loc["review"]
    for channel in [c for c in mix.index if c != "review"]:
        for label in mix.columns:
            share, base = float(mix.at[channel, label]), float(review.get(label, 0.0))
            n = int(counts.get((channel, label), 0))
            r = share / base if base > 0 else None
            if share >= min_share and n >= min_mentions and (r is None or r >= ratio):
                out["silent"].append(ChannelGap(label, channel, share, base, r, n, int(totals[channel])))

    others = mentions[mentions["channel"] != "review"]
    if not others.empty:
        other_share = others.groupby("label").size() / len(others) * 100
        for label in mix.columns:
            base, share = float(review.get(label, 0.0)), float(other_share.get(label, 0.0))
            n_review = int(counts.get(("review", label), 0))
            if base >= min_share and n_review >= min_mentions and (share == 0 or base / share >= ratio):
                out["review_only"].append(
                    ChannelGap(label, "others", share, base, base / share if share else None, n_review, len(others))
                )
    for key in out:
        out[key].sort(key=lambda g: -(g.channel_share if key == "silent" else g.review_share))
    return out


def brand_positions(sessions: pd.DataFrame, index: str = "happiness", min_n: int = 3) -> pd.DataFrame:
    """채널마다 세션 지수를 z-score로 바꾼 뒤 브랜드 평균을 낸다 (브랜드 × 채널).

    0이면 그 채널 평균 수준, +1이면 채널 안에서 1 표준편차 위. 표본이 min_n 미만인 칸은 비운다.
    """
    df = sessions.dropna(subset=["brand", index]).copy()
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("source_type")[index]
    sd = grp.transform("std").replace(0, np.nan)
    df["z"] = (df[index] - grp.transform("mean")) / sd
    table = df.groupby(["brand", "source_type"]).agg(z=("z", "mean"), n=("z", "size")).reset_index()
    table.loc[table["n"] < min_n, "z"] = np.nan
    wide = table.pivot(index="brand", columns="source_type", values="z")
    return wide.dropna(how="all").dropna(axis=1, how="all")


def channel_summary(estimates: pd.DataFrame) -> list[dict]:
    """채널별 추정 CX 지표. estimates는 session_features + apply_estimates 결과."""
    from voclens.cx.estimates import nps_score

    rows = []
    for channel, g in estimates.groupby("source_type"):
        nps, coverage = nps_score(g["nps_group"])
        csat = g["csat_est"].dropna()
        actual = g["rating"].dropna()
        rows.append(
            {
                "channel": channel,
                "label": CHANNEL_LABELS.get(channel, channel),
                "n": len(g),
                "csat_est_mean": float(csat.mean()) if len(csat) else None,
                "csat_pct": float(g["satisfied"].dropna().astype(float).mean() * 100) if len(csat) else None,
                "actual_csat_pct": float((actual >= 4).mean() * 100) if len(actual) else None,
                "ces_mean": float(g["ces_est"].mean()),
                "nps": nps,
                "nps_coverage": coverage,
                "frustration": float(g["frustration"].mean()) if "frustration" in g else None,
                "happiness": float(g["happiness"].mean()) if "happiness" in g else None,
            }
        )
    order = list(CHANNEL_LABELS)
    return sorted(rows, key=lambda r: order.index(r["channel"]) if r["channel"] in order else 99)
