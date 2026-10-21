"""CSAT 추정 모델을 실제 리뷰(네이버 쇼핑)로 다시 검증한다.

배포된 CSAT 모델은 합성 배달 리뷰로 학습한다. 두 가지를 잰다.
  1. 실제 데이터로 학습: 쇼핑 개발용(80%)에서 뽑은 리뷰로 학습하고 평가용(20%) 전체로 평가
  2. 도메인 이전: 합성 배달 리뷰로 학습한 모델(배포 화면과 같은 방식)을 쇼핑 평가용에 그대로 적용
     → 학습한 곳과 다른 데이터에 옮겨 쓰면 얼마나 떨어지는지 (상담·영상에 옮겨 쓰는 상황의 대리 실험)

특징 추출은 배포 코드와 같은 session_features를 쓴다 (분석 결과를 메모리 DB에 저장한 뒤 읽음).
네이버 쇼핑 리뷰의 별점은 1·2·4·5뿐이다 (3점 없음).

    python scripts/csat_shopping.py --dev-limit 40000

결과: experiments/results/csat_shopping.json (원문 없음)
"""

import argparse
import json
import sqlite3

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from voclens.config import REPO_ROOT, load_settings
from voclens.cx.estimates import RidgeRegressor, session_features, train_csat
from voclens.evaluation import analyze_shopping, read_shopping, save_result
from voclens.pipelines.review import ReviewModels
from voclens.store import db

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
SYNTHETIC_DB = REPO_ROOT / "data" / "voclens.db"
OUT = REPO_ROOT / "experiments" / "results" / "csat_shopping.json"


def features(df, settings, models):
    conn = db.connect(":memory:")
    run_id = db.start_run(conn, "review", "naver_shopping", settings, models.versions())
    sessions = analyze_shopping(df, settings, models=models)
    tx = models.taxonomy.taxonomy
    db.save_sessions(conn, sessions, run_id, label_source="keyword",
                     category_of=lambda l: tx.category_of(l) if tx.is_complaint(l) else None)
    out = session_features(conn, review_domain="product")
    conn.close()
    return out[out["source_type"] == "review"].reset_index(drop=True)


def auc(pos, neg) -> float:
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def evaluate(model: RidgeRegressor, test) -> dict:
    y = test["rating"].to_numpy(dtype=float)
    pred = model.predict(test)
    sat = y >= 4
    return {
        "mae": round(float(np.abs(pred - y).mean()), 3),
        "spearman": round(float(spearmanr(pred, y).statistic), 3),
        "auc_satisfied": round(auc(pred[sat], pred[~sat]), 3),
        "csat_pct_actual": round(float(sat.mean() * 100), 1),
        "csat_pct_est": round(float((pred >= model.csat_threshold).mean() * 100), 1),
        "csat_pct_est_raw": round(float((pred >= 4).mean() * 100), 1),
        "threshold": round(float(model.csat_threshold), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-limit", type=int, default=40000, help="학습에 쓸 개발용 리뷰 수 (전체 약 16만)")
    ap.add_argument("--test-limit", type=int, help="빠른 확인용: 평가용에서 뽑을 리뷰 수 (기본: 전체)")
    args = ap.parse_args()
    settings = load_settings()
    models = ReviewModels.default()

    dev = read_shopping(SRC, "dev")
    dev = dev.sample(n=min(args.dev_limit, len(dev)), random_state=42)
    train = features(dev, settings, models)
    test = read_shopping(SRC, "test")
    test = features(test.sample(n=args.test_limit, random_state=42) if args.test_limit else test, settings, models)

    # 1. 실제 데이터로 학습
    model = RidgeRegressor().fit(train, train["rating"])
    model.calibrate_threshold(train, train["rating"])
    valence_only = RidgeRegressor(features=("mean_valence",)).fit(train, train["rating"])
    valence_only.calibrate_threshold(train, train["rating"])
    y = test["rating"].to_numpy(dtype=float)
    constant_mae = float(np.abs(train["rating"].mean() - y).mean())

    # 2. 합성 배달 리뷰로 학습한 모델 (배포 화면과 같은 train_csat)
    synthetic, synthetic_eval = train_csat(session_features(sqlite3.connect(SYNTHETIC_DB)))

    result = {
        "data": {"train_reviews_dev": len(train), "test_reviews": len(test),
                 "test_by_rating": {int(k): int(v) for k, v in test["rating"].value_counts().sort_index().items()}},
        "constant_baseline_mae": round(constant_mae, 3),
        "trained_on_shopping": {**evaluate(model, test), "coefficients": [[k, float(v)] for k, v in model.coefficients()]},
        "valence_only_on_shopping": evaluate(valence_only, test),
        "trained_on_synthetic_delivery": {
            **evaluate(synthetic, test),
            "synthetic_holdout_mae": round(synthetic_eval.mae_model, 3),
            "coefficients": [[k, float(v)] for k, v in synthetic.coefficients()],
        },
    }
    if not args.test_limit:
        save_result(result, OUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
