import numpy as np
import pandas as pd
import pytest

from voclens.cx.compare import brand_positions, channel_gaps, complaint_mix
from voclens.cx.estimates import FEATURES, RidgeRegressor, estimate_ces, nps_group, nps_score, train_csat


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})
    df["rating"] = np.clip(np.round(3 + 1.2 * df["mean_valence"] - 0.6 * df["frustration"] + rng.normal(0, 0.3, n)), 1, 5)
    df["source_type"] = "review"
    df["review_event"] = False
    return df


def test_ridge_learns_signal_and_clips():
    df = _frame()
    model = RidgeRegressor(alpha=1.0).fit(df, df["rating"])
    coefs = dict(model.coefficients())
    assert coefs["mean_valence"] > 0.5 and coefs["frustration"] < -0.2
    assert abs(coefs["stress"]) < 0.2
    pred = model.predict(df)
    assert pred.min() >= 1 and pred.max() <= 5


def test_calibrated_threshold_matches_actual_share():
    df = _frame()
    model = RidgeRegressor().fit(df, df["rating"])
    model.calibrate_threshold(df, df["rating"])
    est_share = (model.predict(df) >= model.csat_threshold).mean()
    assert est_share == pytest.approx((df["rating"] >= 4).mean(), abs=0.05)


def test_train_csat_beats_baselines():
    model, ev = train_csat(_frame(300))
    assert model is not None
    assert ev.mae_model < ev.mae_constant
    assert ev.n_test == 75 and ev.n_train == 225
    assert ev.spearman > 0.5


def test_train_csat_excludes_review_events_and_needs_data():
    df = _frame(30)
    df.loc[:15, "review_event"] = True
    assert train_csat(df) == (None, None)


@pytest.mark.parametrize(
    "row, low, high",
    [
        ({"frustration": 0, "outcome": "resolved"}, 1.0, 1.0),
        ({"frustration": 100, "repeat_contact": True, "waiting": True, "outcome": "pending"}, 7.0, 7.0),
        ({"frustration": 50, "outcome": "transferred"}, 3.0, 4.0),
    ],
)
def test_estimate_ces_range(row, low, high):
    assert low <= estimate_ces(row) <= high


def test_nps_group_priority_and_score():
    assert nps_group({"recommend": True}) == "promoter"
    assert nps_group({"recommend": True, "churn": True}) == "detractor"
    assert nps_group({}) == "unknown"
    assert nps_score(["promoter", "promoter", "detractor", "unknown"]) == (pytest.approx(33.3), 0.75)
    assert nps_score(["unknown"]) == (None, 0.0)


def _mentions(rows):
    out = []
    for channel, label, n in rows:
        out += [{"unit_id": f"{channel}{label}{i}", "channel": channel, "label": label, "brand": "B"} for i in range(n)]
    return pd.DataFrame(out)


def test_complaint_mix_rows_sum_to_100():
    mix = complaint_mix(_mentions([("review", "지연", 6), ("review", "누락", 4), ("call", "지연", 1), ("call", "응대", 3)]))
    assert np.allclose(mix.sum(axis=1), 100)
    assert mix.at["review", "지연"] == pytest.approx(60)


def test_channel_gaps_silent_review_only_and_small_sample():
    m = _mentions([
        ("review", "지연", 40), ("review", "누락", 40), ("review", "맛", 20),
        ("call", "지연", 5), ("call", "응대", 15),          # 응대: 리뷰에 없음 → 조용한 불만
        ("comment", "지연", 3), ("comment", "맛", 1),
    ])
    gaps = channel_gaps(m)
    silent = {(g.label, g.channel): g for g in gaps["silent"]}
    assert ("응대", "call") in silent and silent[("응대", "call")].ratio is None
    assert silent[("응대", "call")].channel_total == 20 and not silent[("응대", "call")].small_sample
    review_only = {g.label for g in gaps["review_only"]}
    assert "누락" in review_only            # 다른 채널 합계에서 0%
    assert "지연" not in review_only


def test_brand_positions_drops_empty_rows():
    df = pd.DataFrame({
        "brand": ["A"] * 4 + ["B"] * 4 + ["C"],
        "source_type": ["review"] * 8 + ["call"],
        "happiness": [80, 70, 75, 85, 20, 30, 25, 35, 50],
    })
    table = brand_positions(df, "happiness")
    assert list(table.index) == ["A", "B"] and list(table.columns) == ["review"]
    assert table.at["A", "review"] > 0 > table.at["B", "review"]
