"""손으로 정한 상수의 민감도 분석.

지수 계산식(indices/compute.py), 개선 과제 우선순위(cx/report.py), CES(cx/estimates.py)의 상수는
데이터로 학습한 값이 아니라 설계 가정이다. 상수를 0.5배~1.5배로 흔들어도 결론(순위, 별점 구분력)이
유지되는지 잰다. 라이브러리 코드는 건드리지 않고 같은 식을 매개변수로 다시 구현하며, 기본값에서
라이브러리 결과와 같은지 먼저 확인한다.

    python scripts/fetch_public_data.py naver_shopping
    python scripts/sensitivity_constants.py              # 쇼핑 평가용 전체
    python scripts/sensitivity_constants.py --limit 5000 # 빠르게

결과: experiments/results/sensitivity_constants.json (원문 없음)
"""

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

from voclens.config import REPO_ROOT, load_settings
from voclens.cx.estimates import estimate_ces, session_features
from voclens.cx.report import load_report_data, pain_points
from voclens.evaluation import analyze_shopping, read_shopping, save_result
from voclens.indices.compute import engagement_index, unit_scores
from voclens.pipelines.review import ReviewModels
from voclens.schema import Unit
from voclens.text.taxonomy import Taxonomy

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
DB = REPO_ROOT / "data" / "voclens.db"
OUT = REPO_ROOT / "experiments" / "results" / "sensitivity_constants.json"

# 기본값 = 라이브러리에 적힌 값
UNIT = {"w_repeat": 0.40, "w_wait": 0.25, "w_churn": 0.15, "w_neg": 0.40,
        "d_stress": 0.6, "d_happy": 0.8, "d_neg": 0.5, "arousal_default": 0.3}
ENGAGE = {"w_chars": 0.35, "w_aspects": 0.35, "w_intent": 0.30, "chars_full": 150, "aspects_full": 3}
PRIORITY = {"base": 0.5, "trend": 0.5}
CES = {"w_frus": 0.6, "w_repeat": 0.2, "w_wait": 0.1, "w_unresolved": 0.1}
QUADRANTS = {(True, True): "Quick Win", (True, False): "핵심 과제", (False, True): "쉬운 개선", (False, False): "후순위"}

LOW, HIGH = 0.5, 1.5
DRAWS = 500


# ---------- 같은 식, 매개변수판 ----------

def unit_frame(units: pd.DataFrame, p: dict) -> pd.DataFrame:
    """units: valence, arousal(결측 가능), repeat, wait, churn → stress·happiness·negative_affect·frustration."""
    v = units["valence"].fillna(0.0).to_numpy()
    a = units["arousal"].fillna(p["arousal_default"]).to_numpy()
    neg = np.maximum(0.0, -v)
    frus = (p["w_repeat"] * units["repeat"] + p["w_wait"] * units["wait"]
            + p["w_churn"] * units["churn"] + p["w_neg"] * neg)
    clip = lambda x: np.clip(x, 0.0, 1.0)  # noqa: E731
    return pd.DataFrame({
        "stress": clip(neg * a / p["d_stress"]),
        "happiness": clip(np.maximum(0.0, v) / p["d_happy"]),
        "negative_affect": clip(neg * (1 - a) / p["d_neg"]),
        "frustration": clip(frus.to_numpy()),
    }, index=units.index)


def engagement(sessions: pd.DataFrame, p: dict) -> np.ndarray:
    """sessions: chars, aspects, intent (세션 단위)."""
    score = (p["w_chars"] * np.minimum(sessions["chars"] / p["chars_full"], 1)
             + p["w_aspects"] * np.minimum(sessions["aspects"] / p["aspects_full"], 1)
             + p["w_intent"] * sessions["intent"])
    return np.clip(score.to_numpy(), 0, 1) * 100


def priorities(items: list[dict], unit_scores_df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """items: label, share, impact, trend, difficulty, unit_ids. 강도는 바뀐 단위 점수로 다시 계산한다."""
    rows = []
    for i in items:
        s = unit_scores_df.loc[i["unit_ids"], ["stress", "frustration", "negative_affect"]].max(axis=1)
        rows.append({"label": i["label"], "share": i["share"], "intensity": float(s.mean()),
                     "impact": i["impact"], "trend": i["trend"], "difficulty": i["difficulty"]})
    df = pd.DataFrame(rows)
    boost = 1 + p["trend"] * df["trend"].fillna(0.0).clip(lower=0.0)
    raw = (df["share"] / df["share"].max()) * (p["base"] + df["intensity"]) * boost * (1 + df["impact"])
    df["priority"] = 100 * raw / raw.max()
    high = df["priority"] >= df["priority"].median()
    df["quadrant"] = [QUADRANTS[(h, d <= 1)] for h, d in zip(high, df["difficulty"])]
    return df.sort_values("priority", ascending=False).reset_index(drop=True)


def ces(calls: pd.DataFrame, p: dict) -> np.ndarray:
    unresolved = calls["outcome"].isin(["pending", "transferred"])
    effort = (p["w_frus"] * calls["frustration"].fillna(0) / 100 + p["w_repeat"] * calls["repeat_contact"].astype(bool)
              + p["w_wait"] * calls["waiting"].astype(bool) + p["w_unresolved"] * unresolved)
    return (1 + 6 * effort.clip(0, 1)).to_numpy()


# ---------- 흔들기 ----------

def variants(base: dict, rng: np.random.Generator, skip=()):
    """(이름, 매개변수): 하나씩 0.5배·1.5배, 그다음 전부 동시에 무작위 0.5~1.5배."""
    keys = [k for k in base if k not in skip]
    for k in keys:
        for f in (LOW, HIGH):
            yield f"{k}×{f}", {**base, k: base[k] * f}
    for n in range(DRAWS):
        yield f"joint#{n}", {**base, **{k: base[k] * rng.uniform(LOW, HIGH) for k in keys}}


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"min": round(float(arr.min()), 4), "p05": round(float(np.percentile(arr, 5)), 4),
            "median": round(float(np.median(arr)), 4)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(양성 점수 > 음성 점수), 동점은 절반."""
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


# ---------- 데이터 ----------

def flatten(sessions, rating_key="rating") -> tuple[pd.DataFrame, pd.DataFrame]:
    urows, srows = [], []
    for s in sessions:
        for u in s.units:
            sig = u.cx_signals
            urows.append({"unit_id": u.unit_id, "session_id": s.session_id, "valence": u.valence, "arousal": u.arousal,
                          "repeat": bool(sig.get("repeat_contact")), "wait": bool(sig.get("waiting")),
                          "churn": bool(sig.get("churn_intent") or sig.get("not_recommend")),
                          "labels": list(u.complaint_categories)})
        srows.append({"session_id": s.session_id, "rating": s.meta.get(rating_key),
                      "chars": sum(len(u.text) for u in s.units), "aspects": len({u.aspect for u in s.units if u.aspect}),
                      "intent": any(u.cx_signals.get("repurchase_intent") or u.cx_signals.get("recommend_intent") for u in s.units),
                      "churn": any(u.cx_signals.get("churn_intent") or u.cx_signals.get("not_recommend") for u in s.units),
                      "repurchase": any(u.cx_signals.get("repurchase_intent") for u in s.units),
                      "_engage_lib": engagement_index(s.units)})
    return pd.DataFrame(urows).set_index("unit_id"), pd.DataFrame(srows).set_index("session_id")


def check_unit_formula(units: pd.DataFrame, sample: list[Unit]) -> None:
    mine = unit_frame(units.loc[[u.unit_id for u in sample]], UNIT)
    for u in sample:
        lib = unit_scores(u)
        for k, v in lib.items():
            assert abs(mine.loc[u.unit_id, k] - v) < 1e-3, (u.unit_id, k, mine.loc[u.unit_id, k], v)


def session_indices(units: pd.DataFrame, sessions: pd.DataFrame, p_unit: dict, p_eng: dict) -> pd.DataFrame:
    scores = unit_frame(units, p_unit)
    out = scores.groupby(units["session_id"]).mean() * 100
    out = out.reindex(sessions.index)
    out["engagement"] = engagement(sessions, p_eng)
    return out


def shopping_items(units: pd.DataFrame, sessions: pd.DataFrame, tx: Taxonomy, min_reviews: int = 40) -> list[dict]:
    """pain_points와 같은 방식(추세 없음: 쇼핑 리뷰에는 날짜가 없다)."""
    churn_all, rep_all = sessions["churn"].mean(), sessions["repurchase"].mean()
    items = []
    exploded = units[["session_id", "labels"]].explode("labels").dropna()
    exploded = exploded[exploded["labels"].map(tx.is_complaint)]
    for label, g in exploded.groupby("labels"):
        sids = g["session_id"].unique()
        if len(sids) < min_reviews:
            continue
        f = sessions.loc[sids]
        impact = max(0.0, f["churn"].mean() - churn_all) + max(0.0, rep_all - f["repurchase"].mean())
        items.append({"label": label, "share": len(sids) / len(sessions), "impact": float(impact), "trend": None,
                      "difficulty": tx.difficulty_of(label), "unit_ids": list(g.index)})
    return items


def priority_stability(items: list[dict], units: pd.DataFrame, rng, top_k: int = 3) -> dict:
    base_scores = unit_frame(units, UNIT)
    base = priorities(items, base_scores, PRIORITY)
    base_order = list(base["label"])
    base_quad = dict(zip(base["label"], base["quadrant"]))
    top = set(base_order[:top_k])
    rho, top_same, first_same, quad_changed, worst = [], [], [], [], None
    for name, p in variants({**UNIT, **PRIORITY}, rng):
        pu = {k: p[k] for k in UNIT}
        pp = {k: p[k] for k in PRIORITY}
        df = priorities(items, unit_frame(units, pu), pp)
        order = list(df["label"])
        r = spearmanr([base_order.index(l) for l in order], range(len(order))).statistic
        rho.append(r)
        top_same.append(set(order[:top_k]) == top)
        first_same.append(order[0] == base_order[0])
        changed = float(np.mean([base_quad[l] != q for l, q in zip(df["label"], df["quadrant"])]))
        quad_changed.append(changed)
        if worst is None or r < worst[1]:
            worst = (name, r)
    return {
        "types": len(items),
        "baseline_order": base_order,
        "baseline_quadrant": base_quad,
        "rank_spearman": summarize(rho),
        f"top{top_k}_set_unchanged": round(float(np.mean(top_same)), 3),
        "first_unchanged": round(float(np.mean(first_same)), 3),
        "quadrant_changed_share": summarize(quad_changed) | {"mean": round(float(np.mean(quad_changed)), 4)},
        "worst_variant": {"name": worst[0], "spearman": round(float(worst[1]), 4)},
    }


# ---------- 실험 ----------

def experiment_shopping(limit: int | None, rng) -> tuple[dict, pd.DataFrame, pd.DataFrame, Taxonomy]:
    test = read_shopping(SRC, "test")
    if limit:
        test = test.sample(n=limit, random_state=42)
    models = ReviewModels.default()
    sessions_obj = list(analyze_shopping(test, load_settings(), models=models))
    units, sessions = flatten(sessions_obj)
    check_unit_formula(units, [u for s in sessions_obj[:300] for u in s.units])
    assert np.allclose(engagement(sessions, ENGAGE), sessions["_engage_lib"], atol=0.01)

    low = (sessions["rating"] <= 2).to_numpy()
    high = (sessions["rating"] >= 4).to_numpy()
    base = session_indices(units, sessions, UNIT, ENGAGE)
    # 행복·몰입은 고득점에서 높아야 하고, 나머지는 저득점에서 높아야 한다
    direction = {"happiness": 1, "engagement": 1, "stress": -1, "negative_affect": -1, "frustration": -1}

    def aucs(ix):
        out = {}
        for k, d in direction.items():
            x = ix[k].fillna(0).to_numpy()
            out[k] = auc(x[high], x[low]) if d > 0 else auc(x[low], x[high])
        return out

    base_auc = aucs(base)
    rho = {k: [] for k in direction}
    auc_all = {k: [] for k in direction}
    for name, p in variants({**UNIT, **ENGAGE}, rng):
        pu = {k: p[k] for k in UNIT}
        pe = {k: p[k] for k in ENGAGE}
        ix = session_indices(units, sessions, pu, pe)
        for k in direction:
            rho[k].append(spearmanr(base[k].fillna(0), ix[k].fillna(0)).statistic)
        for k, v in aucs(ix).items():
            auc_all[k].append(v)
    result = {
        "reviews": len(sessions),
        "units": len(units),
        "indices": {k: {"baseline_auc": round(base_auc[k], 4), "auc": summarize(auc_all[k]),
                        "rank_spearman_vs_baseline": summarize(rho[k])} for k in direction},
    }
    return result, units, sessions, models.taxonomy.taxonomy


def experiment_synthetic_priority(rng) -> dict:
    conn = sqlite3.connect(DB)
    data = load_report_data(conn)
    tx = Taxonomy.load()
    lib = pain_points(data, tx, top_n=100)
    complaints = data.labels[data.labels["label"].map(tx.is_complaint)].merge(data.units, on="unit_id")
    units = data.units.set_index("unit_id")
    arousal = pd.read_sql_query("SELECT unit_id, arousal FROM units WHERE source_type = 'review'", conn).set_index("unit_id")
    signals = units["cx_signals_json"].map(json.loads)
    frame = pd.DataFrame({
        "session_id": units["session_id"], "valence": units["valence"], "arousal": arousal["arousal"].reindex(units.index),
        "repeat": signals.map(lambda d: bool(d.get("repeat_contact"))), "wait": signals.map(lambda d: bool(d.get("waiting"))),
        "churn": signals.map(lambda d: bool(d.get("churn_intent") or d.get("not_recommend"))),
    })
    items = [{**i, "unit_ids": list(complaints.loc[complaints["label"] == i["label"], "unit_id"])} for i in lib]
    mine = priorities(items, unit_frame(frame, UNIT), PRIORITY)
    lib_order = [i["label"] for i in lib]
    assert list(mine["label"]) == lib_order, (list(mine["label"]), lib_order)
    assert np.allclose(mine.set_index("label").loc[lib_order, "priority"], [i["priority"] for i in lib], atol=0.01)
    return priority_stability(items, frame, rng)


def experiment_calls(rng) -> dict:
    conn = sqlite3.connect(DB)
    df = session_features(conn)
    calls = df[df["source_type"] == "call"].reset_index(drop=True)
    lib = calls.apply(lambda r: estimate_ces(r.to_dict()), axis=1).to_numpy()
    base = ces(calls, CES)
    assert np.allclose(base, lib, atol=0.01)
    hardest = int(np.argmax(base))
    taus, hardest_same, order_same = [], [], []
    for _, p in variants(CES, rng):
        x = ces(calls, p)
        taus.append(kendalltau(base, x).statistic)
        hardest_same.append(int(np.argmax(x)) == hardest)
        order_same.append(list(np.argsort(-x)) == list(np.argsort(-base)))
    return {
        "calls": len(calls),
        "baseline_ces": [round(float(v), 2) for v in base],
        "kendall_tau_vs_baseline": summarize(taus),
        "hardest_call_unchanged": round(float(np.mean(hardest_same)), 3),
        "full_order_unchanged": round(float(np.mean(order_same)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="쇼핑 평가용에서 뽑을 리뷰 수 (기본: 전체)")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    shopping, units, sessions, tx = experiment_shopping(args.limit, rng)
    result = {
        "method": f"상수를 하나씩 ×{LOW}·×{HIGH}, 그리고 전부 동시에 무작위 ×U({LOW},{HIGH}) {DRAWS}회. 기본값은 라이브러리와 일치 확인",
        "constants": {"unit": UNIT, "engagement": ENGAGE, "priority": PRIORITY, "ces": CES},
        "A_indices_shopping": shopping,
        "B_priority_synthetic_delivery": experiment_synthetic_priority(rng),
        "B_priority_shopping": priority_stability(shopping_items(units, sessions, tx), units, rng),
        "C_ces_synthetic_calls": experiment_calls(rng),
    }
    save_result(result, OUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
