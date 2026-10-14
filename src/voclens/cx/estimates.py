"""CX 지표 추정: CSAT · CES · NPS 성향 (docs/analysis_design.md 6.2).

CSAT
  리뷰 별점을 CSAT의 대리 정답으로 쓴다. 세션 특징(감정, 지수, CX 신호)으로 별점을 예측하는
  릿지 회귀를 리뷰로 학습하고, 같은 특징을 뽑을 수 있는 상담·영상에 적용한다.
  리뷰 이벤트 리뷰는 별점이 감정과 무관하게 높아서 학습에서 뺀다.
  평가는 학습에 쓰지 않은 리뷰(25%)로 하고, 두 기준선과 비교한다:
    상수 기준선(학습 데이터 평균 별점), Valence 단일 특징 회귀.
  상담·영상에 적용한 값은 도메인이 달라 오차가 클 수 있어 "추정"으로만 표시한다.

CES (1 = 쉬움 ~ 7 = 어려움)
  좌절 지수 + 반복 문의 + 대기 + 미해결(보류·이관)로 계산한 노력 점수를 1~7로 옮긴다.
  공식 CES 설문 점수가 아니다.

NPS 성향
  추천·재주문 의사 → 추천 고객, 이탈·비추천 의사 → 비추천 고객,
  둘 다 없으면 "판단 불가"(감정만으로 추천 여부를 추측하지 않는다). 둘 다 있으면 비추천을 우선한다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from voclens.text.taxonomy import normalize_domain

FEATURES = (
    "mean_valence", "min_valence", "last_valence",
    "stress", "happiness", "negative_affect", "frustration", "peak_end",
    "n_complaints", "churn", "not_recommend", "repurchase", "recommend",
)
SIGNALS = ("churn_intent", "not_recommend", "repurchase_intent", "recommend_intent", "repeat_contact", "waiting")
# 채널별로 지수를 계산할 때 쓰는 발화 (상담은 고객, 영상은 크리에이터·게스트, 댓글은 따로)
TARGET_ROLES = {"review": None, "call": ("customer",), "video": ("creator", "guest")}


def _target_units(units: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=units.index)
    for source_type, roles in TARGET_ROLES.items():
        if roles:
            mask = units["source_type"] == source_type
            keep &= ~mask | units["speaker_role"].isin(roles)
    return units[keep]


def session_features(conn: sqlite3.Connection, review_domain: str | None = "delivery") -> pd.DataFrame:
    """세션마다 CSAT·CES·NPS 추정에 쓰는 특징을 만든다 (리뷰·상담·영상 공통).

    domain: 상담·영상은 메타데이터의 domain, 리뷰는 review_domain(설정값)을 쓴다.
    """
    sessions = pd.read_sql_query(
        "SELECT s.session_id, s.source_type, s.flags_json, s.peak_end, src.brand, src.rating, src.meta_json"
        " FROM sessions s JOIN sources src ON src.source_id = s.source_id",
        conn,
    )
    if sessions.empty:
        return sessions
    units = pd.read_sql_query(
        "SELECT unit_id, session_id, source_type, speaker_role, ord, valence, cx_signals_json FROM units", conn
    )
    units = _target_units(units).sort_values(["session_id", "ord"])
    signals = units["cx_signals_json"].map(json.loads)
    for name in SIGNALS:
        units[name] = signals.map(lambda d, n=name: bool(d.get(n, False)))

    labels = pd.read_sql_query(
        "SELECT l.unit_id FROM unit_labels l WHERE l.label_type = 'complaint' AND l.category IS NOT NULL", conn
    )
    units["complaint"] = units["unit_id"].isin(set(labels["unit_id"]))

    g = units.groupby("session_id")
    feats = pd.DataFrame(
        {
            "mean_valence": g["valence"].mean(),
            "min_valence": g["valence"].min(),
            "last_valence": g["valence"].last(),
            "n_complaints": g["complaint"].sum(),
            **{name: g[name].any() for name in SIGNALS},
        }
    ).reset_index()

    indices = pd.read_sql_query("SELECT session_id, index_name, raw FROM indices", conn)
    wide = indices.pivot(index="session_id", columns="index_name", values="raw").reset_index()

    df = sessions.merge(feats, on="session_id", how="left").merge(wide, on="session_id", how="left")
    df = df.rename(
        columns={"churn_intent": "churn", "repurchase_intent": "repurchase", "recommend_intent": "recommend"}
    )
    meta = df["meta_json"].fillna("{}").map(json.loads)
    df["outcome"] = meta.map(lambda m: m.get("outcome"))
    df["domain"] = meta.map(lambda m: normalize_domain(m.get("domain")))
    df.loc[df["source_type"] == "review", "domain"] = normalize_domain(review_domain)
    df["review_event"] = df["flags_json"].fillna("").str.contains("review_event")
    for col in ("churn", "not_recommend", "repurchase", "recommend", "repeat_contact", "waiting"):
        df[col] = df[col].fillna(False).astype(bool)
    return df.drop(columns=["flags_json", "meta_json"])


def _matrix(df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    return df[list(features)].astype(float).fillna(0.0).to_numpy()


@dataclass
class RidgeRegressor:
    """표준화 + 릿지 회귀 (scikit-learn 없이). 예측은 1~5로 자른다."""

    alpha: float = 1.0
    features: tuple = FEATURES
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    # 추정 별점이 이 값 이상이면 "만족(4점 이상)"으로 센다. 회귀는 예측이 평균 쪽으로 몰려서
    # 4.0을 그대로 쓰면 만족 비율이 낮게 나온다. 학습 데이터의 실제 만족 비율과 같아지도록 잡는다.
    csat_threshold: float = 4.0

    def fit(self, df: pd.DataFrame, y) -> RidgeRegressor:
        X = _matrix(df, self.features)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        Z = (X - self.mean_) / self.scale_
        self.intercept_ = float(y.mean())
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - self.intercept_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        Z = (_matrix(df, self.features) - self.mean_) / self.scale_
        return np.clip(Z @ self.coef_ + self.intercept_, 1.0, 5.0)

    def calibrate_threshold(self, df: pd.DataFrame, y) -> float:
        actual_share = float((np.asarray(y, dtype=float) >= 4).mean())
        pred = self.predict(df)
        self.csat_threshold = float(np.quantile(pred, 1 - actual_share)) if 0 < actual_share < 1 else 4.0
        return self.csat_threshold

    def coefficients(self) -> list[tuple[str, float]]:
        return sorted(zip(self.features, self.coef_.round(3)), key=lambda kv: -abs(kv[1]))


@dataclass
class CsatEvaluation:
    n_train: int
    n_test: int
    mae_model: float
    mae_valence_only: float
    mae_constant: float
    spearman: float
    csat_pct_actual: float = 0.0       # 평가 데이터의 실제 만족 비율 (별점 4 이상)
    csat_pct_est: float = 0.0          # 같은 데이터의 추정 만족 비율 (보정한 임계값 기준)
    csat_pct_est_raw: float = 0.0      # 보정 없이 추정 별점 4.0 이상으로 셌을 때
    coefficients: list = field(default_factory=list)

    @property
    def improvement_vs_valence(self) -> float:
        return (self.mae_valence_only - self.mae_model) / self.mae_valence_only if self.mae_valence_only else 0.0


def train_csat(df: pd.DataFrame, seed: int = 42, test_size: float = 0.25, alpha: float = 1.0) -> tuple[RidgeRegressor | None, CsatEvaluation | None]:
    """별점이 있는 일반 리뷰로 학습·평가한다. 평가 후 전체 데이터로 다시 학습한 모델을 돌려준다."""
    train_df = df[(df["source_type"] == "review") & df["rating"].notna() & ~df["review_event"]]
    if len(train_df) < 20:
        return None, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_df))
    n_test = max(5, int(len(train_df) * test_size))
    test, train = train_df.iloc[order[:n_test]], train_df.iloc[order[n_test:]]

    model = RidgeRegressor(alpha=alpha).fit(train, train["rating"])
    model.calibrate_threshold(train, train["rating"])
    valence_only = RidgeRegressor(alpha=alpha, features=("mean_valence",)).fit(train, train["rating"])
    pred = model.predict(test)
    y = test["rating"].to_numpy(dtype=float)
    evaluation = CsatEvaluation(
        n_train=len(train),
        n_test=len(test),
        mae_model=float(np.abs(pred - y).mean()),
        mae_valence_only=float(np.abs(valence_only.predict(test) - y).mean()),
        mae_constant=float(np.abs(train["rating"].mean() - y).mean()),
        spearman=float(spearmanr(pred, y).statistic) if len(set(y)) > 1 else 0.0,
        csat_pct_actual=float((y >= 4).mean() * 100),
        csat_pct_est=float((pred >= model.csat_threshold).mean() * 100),
        csat_pct_est_raw=float((pred >= 4).mean() * 100),
    )
    final = RidgeRegressor(alpha=alpha).fit(train_df, train_df["rating"])
    final.calibrate_threshold(train_df, train_df["rating"])
    evaluation.coefficients = final.coefficients()
    return final, evaluation


def estimate_ces(row) -> float:
    """1(쉬움) ~ 7(어려움). 좌절 지수 60%, 반복 문의 20%, 대기 10%, 미해결 10%."""
    frustration = (row.get("frustration") or 0.0) / 100
    unresolved = row.get("outcome") in ("pending", "transferred")
    effort = 0.6 * frustration + 0.2 * bool(row.get("repeat_contact")) + 0.1 * bool(row.get("waiting")) + 0.1 * unresolved
    return round(1 + 6 * min(1.0, max(0.0, effort)), 2)


def nps_group(row) -> str:
    if row.get("churn") or row.get("not_recommend"):
        return "detractor"
    if row.get("recommend") or row.get("repurchase"):
        return "promoter"
    return "unknown"


def nps_score(groups) -> tuple[float | None, float]:
    """(NPS, 판단 가능 비율). 판단 가능한 세션만으로 계산한다."""
    groups = list(groups)
    known = [g for g in groups if g != "unknown"]
    if not known:
        return None, 0.0
    nps = 100 * (known.count("promoter") - known.count("detractor")) / len(known)
    return round(nps, 1), round(len(known) / len(groups), 3)


def apply_estimates(df: pd.DataFrame, model: RidgeRegressor | None) -> pd.DataFrame:
    df = df.copy()
    df["csat_est"] = model.predict(df).round(2) if model is not None else np.nan
    df["satisfied"] = (df["csat_est"] >= model.csat_threshold) if model is not None else np.nan
    df["ces_est"] = df.apply(lambda r: estimate_ces(r.to_dict()), axis=1)
    df["nps_group"] = df.apply(lambda r: nps_group(r.to_dict()), axis=1)
    return df


def save_estimates(conn: sqlite3.Connection, df: pd.DataFrame, model_name: str) -> None:
    with conn:
        conn.execute("DELETE FROM estimates")
        conn.executemany(
            "INSERT INTO estimates (session_id, csat_est, ces_est, nps_group, model) VALUES (?, ?, ?, ?, ?)",
            [
                (r.session_id, None if pd.isna(r.csat_est) else float(r.csat_est), float(r.ces_est), r.nps_group, model_name)
                for r in df.itertuples()
            ],
        )
