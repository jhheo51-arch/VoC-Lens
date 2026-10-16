"""학습 감정 모델용 주제 표현 임계치(topic_max_valence)를 개발용 데이터에서 정한다. 평가 세트는 쓰지 않는다.

두 가지 방식
  별점 대리 지표 (기본)  5점 리뷰 불만 비율을 감정 사전 이하로 묶고 1·2점 리뷰 불만 비율 최대
                         → 평가 세트에서 실패했다 (5점 리뷰에도 실제 불만 문장이 있어서). 기록용으로 남긴다
  개발용 정답 (--gold)   개발용 리뷰에서 뽑아 라벨링한 절에서 불만 여부 F1이 가장 높은 값

    python scripts/tune_topic_threshold.py --emotion-model models/emotion-kote-small
    python scripts/tune_topic_threshold.py --emotion-model models/emotion-kote-small --gold labeling/dev/naver_shopping_dev_labels.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from voclens.config import REPO_ROOT, load_settings
from voclens.evaluation import _complaints, analyze_shopping, load_emotion_model, read_shopping, save_result
from voclens.ingest.review_csv import ReviewRecord
from voclens.pipelines.review import ReviewModels, build_session
from voclens.text.taxonomy import KeywordTaxonomyClassifier

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
GRID = [round(x, 2) for x in np.arange(0.2, -0.61, -0.05)]


class CachedEmotion:
    """한 번 추론한 결과를 문장별로 재사용한다 (임계치만 바꿔 여러 번 돌릴 때)."""

    def __init__(self, model, texts):
        uniq = sorted(set(texts))
        self.cache = dict(zip(uniq, model.score_many(uniq)))
        self.name, self.model = model.name, model

    def score_many(self, texts):
        return [self.cache[t] for t in texts]

    predict = None   # 아래에서 원래 모델의 predict 로직을 빌려 쓴다


def rates(df, settings, models) -> dict:
    tx = models.taxonomy.taxonomy
    by = {}
    for s in analyze_shopping(df, settings, models=models):
        by.setdefault(int(s.meta["rating"]), []).append(any(_complaints(u.complaint_categories, tx) for u in s.units))
    out = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    out["neg"] = float(np.mean(by[1] + by[2]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion-model", required=True)
    ap.add_argument("--per-rating", type=int, default=1000)
    ap.add_argument("--gold", help="개발용 절 정답 JSON. 주면 별점 대신 이 정답의 불만 여부 F1로 고른다")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "topic_threshold_tuning.json"))
    args = ap.parse_args()
    settings = load_settings()
    if args.gold:
        return tune_on_gold(args, settings)
    dev = read_shopping(SRC, "dev")
    sample = pd.concat([g.sample(args.per_rating, random_state=21) for _, g in dev.groupby("rating")])

    base = rates(sample, settings, ReviewModels.default())
    print(f"감정 사전: 1·2점 {base['neg']:.1%} · 5점 {base[5]:.1%}")

    model = load_emotion_model(args.emotion_model)
    texts = [u.clean_text or u.text for r in sample.itertuples()
             for u in build_session(ReviewRecord(source_id=f"ns{r.line}", text=r.text, rating=float(r.rating))).units]
    cached = CachedEmotion(model, texts)
    cached.predict = lambda units: type(model).predict(cached, units)   # 같은 predict 로직, 추론만 캐시에서

    rows = []
    for t in GRID:
        models = ReviewModels.default()
        models.emotion = cached
        models.taxonomy = KeywordTaxonomyClassifier(topic_max_valence=t)
        r = rates(sample, settings, models)
        rows.append({"topic_max_valence": float(t), "neg_rate": round(r["neg"], 4), "five_star_rate": round(r[5], 4), "four_star_rate": round(r[4], 4)})
        print(f"임계치 {t:+.2f}: 1·2점 {r['neg']:.1%} · 4점 {r[4]:.1%} · 5점 {r[5]:.1%}")
    ok = [r for r in rows if r["five_star_rate"] <= base[5]]
    chosen = max(ok, key=lambda r: r["neg_rate"]) if ok else min(rows, key=lambda r: r["five_star_rate"])
    print(f"선택: {chosen}")
    save_result({"emotion_model": model.name, "split": "dev", "per_rating": args.per_rating,
                 "lexicon": {"neg_rate": round(base["neg"], 4), "five_star_rate": round(base[5], 4)},
                 "grid": rows, "chosen": chosen,
                 "rule": "5점 리뷰 불만 비율이 감정 사전 이하인 값 중 1·2점 리뷰 불만 비율이 가장 높은 값"}, args.out)


def tune_on_gold(args, settings) -> None:
    from voclens.evaluation import evaluate_gold, predict_units, read_gold

    gold = read_gold(args.gold)
    assert all(read_shopping_split(u) == "dev" for u in gold["unit_id"]), "개발용 정답에는 평가용(test) 리뷰가 없어야 한다"
    shop = read_shopping(SRC)
    model = load_emotion_model(args.emotion_model)
    lines = sorted({int(u.split("#")[0].removeprefix("review:ns")) for u in gold["unit_id"]})
    texts = [u.clean_text or u.text for r in shop[shop["line"].isin(lines)].itertuples()
             for u in build_session(ReviewRecord(source_id=f"ns{r.line}", text=r.text, rating=float(r.rating))).units]
    cached = CachedEmotion(model, texts)
    cached.predict = lambda units: type(model).predict(cached, units)
    ids = gold["unit_id"].tolist()
    lex = evaluate_gold(gold, predict_units(ids, shop, settings))["any_complaint"]
    print(f"감정 사전: 정밀도 {lex['precision']} · 재현율 {lex['recall']} · F1 {lex['f1']}")
    rows = []
    for t in GRID:
        models = ReviewModels.default()
        models.emotion = cached
        models.taxonomy = KeywordTaxonomyClassifier(topic_max_valence=t)
        a = evaluate_gold(gold, predict_units(ids, shop, settings, models=models))["any_complaint"]
        rows.append({"topic_max_valence": float(t), **{k: a[k] for k in ("precision", "recall", "f1")}})
        print(f"임계치 {t:+.2f}: 정밀도 {a['precision']} · 재현율 {a['recall']} · F1 {a['f1']}")
    chosen = max(rows, key=lambda r: (r["f1"] or 0, -abs(r["topic_max_valence"])))
    print(f"선택: {chosen}")
    save_result({"emotion_model": model.name, "split": "dev", "gold": str(args.gold), "n": int((~gold["hold"]).sum()),
                 "lexicon": {k: lex[k] for k in ("precision", "recall", "f1")}, "grid": rows, "chosen": chosen,
                 "rule": "개발용 절 정답에서 불만 여부 F1이 가장 높은 값 (같으면 0에 가까운 값)"}, args.out)


def read_shopping_split(unit_id: str) -> str:
    from voclens.evaluation import split_of

    return split_of(int(unit_id.split("#")[0].removeprefix("review:ns")))


if __name__ == "__main__":
    main()
