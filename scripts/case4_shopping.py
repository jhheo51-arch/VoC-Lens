"""케이스 4 집계: 네이버 쇼핑 리뷰 평가용 20%의 불만 유형별 크기·치명도, 재구매 고객 비교.

    python scripts/fetch_public_data.py naver_shopping
    python scripts/case4_shopping.py

결과는 원문 없이 수치만 experiments/results/case4_*.json 에 저장한다.
"""

from collections import Counter

import pandas as pd

from voclens.config import REPO_ROOT, load_settings
from voclens.evaluation import _complaints, analyze_shopping, read_shopping, save_result
from voclens.pipelines.review import ReviewModels

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
OUT = REPO_ROOT / "experiments" / "results"
MIN_REVIEWS = 40


def main() -> None:
    test = read_shopping(SRC, "test")
    models = ReviewModels.default()
    tx = models.taxonomy.taxonomy
    repeat = dict(zip("ns" + test["line"].astype(str), test["repeat_buyer"]))
    rows = []
    for s in analyze_shopping(test, load_settings(), models=models):
        labels = {l for u in s.units for l in _complaints(u.complaint_categories, tx)}
        rows.append({
            "rating": int(s.meta["rating"]),
            "repeat": bool(repeat[s.source_id]),
            "labels": labels,
            "unknown_only": not labels and any(tx.unknown_label in u.complaint_categories for u in s.units),
        })
    r = pd.DataFrame(rows)
    neg, pos = r[r["rating"] <= 2], r[r["rating"] >= 4]

    types = []
    for label in sorted({l for ls in r["labels"] for l in ls}):
        has = r["labels"].map(lambda ls, l=label: l in ls)
        n_neg, n_pos = int((has & (r["rating"] <= 2)).sum()), int((has & (r["rating"] >= 4)).sum())
        if has.sum() < MIN_REVIEWS:
            continue
        types.append({
            "type": label, "category": tx.category_of(label), "difficulty": tx.difficulty_of(label), "reviews": int(has.sum()),
            "share_of_negative": round(n_neg / len(neg), 3),
            "one_star_among_negative": round(int((has & (r["rating"] == 1)).sum()) / max(1, n_neg), 3),
            "share_in_positive_reviews": round(n_pos / int(has.sum()), 3),
        })
    categories = Counter(c for ls in neg["labels"] for c in {tx.category_of(x) for x in ls})
    segments = {}
    for name, flag in (("repeat", True), ("new", False)):
        g = neg[neg["repeat"] == flag]
        mix = Counter(l for ls in g["labels"] for l in ls)
        segments[name] = {"negative_reviews": len(g), "one_star_share": round(float((g["rating"] == 1).mean()), 3),
                          "top_types": {k: round(v / len(g), 3) for k, v in mix.most_common(8)}}
    summary = {
        "reviews": len(r), "by_rating": {int(k): int(v) for k, v in r["rating"].value_counts().sort_index().items()},
        "negative_with_type": round(float(neg["labels"].map(bool).mean()), 3),
        "negative_unknown_only": round(float(neg["unknown_only"].mean()), 3),
        "one_star_among_negative": round(float((neg["rating"] == 1).mean()), 3),
        "category_share_of_negative": {k: round(v / len(neg), 3) for k, v in categories.most_common()},
        "segments": segments,
    }
    save_result(sorted(types, key=lambda t: -t["share_of_negative"]), OUT / "case4_type_table.json")
    save_result(summary, OUT / "case4_summary.json")
    print(pd.DataFrame(types).sort_values("share_of_negative", ascending=False).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
