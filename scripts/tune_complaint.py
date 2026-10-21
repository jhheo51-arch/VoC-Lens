"""학습한 불만 분류기의 합치는 방식(union / model)과 임계치를 개발용 절 정답에서 고른다. 평가 세트는 쓰지 않는다.

선택 규칙(미리 정함): 불만 여부 F1이 가장 높은 조합. 같으면 불만 유형 micro F1이 높은 쪽.
감정 점수는 감정 사전과 학습 감정 모델(--emotion-model) 둘 다 시험한다.

    python scripts/tune_complaint.py --complaint-model models/complaint-weak-small --emotion-model models/emotion-kote-small
"""

import argparse

from voclens.config import REPO_ROOT, load_settings
from voclens.evaluation import evaluate_gold, load_emotion_model, predict_units, read_gold, read_shopping, save_result, split_of
from voclens.pipelines.review import ReviewModels
from voclens.text.complaint_model import TransformerComplaintClassifier

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
GOLD = REPO_ROOT / "labeling" / "dev" / "naver_shopping_dev_labels.json"
# 0.9가 탐색 범위 끝이었으므로(평가 문서 5절) 0.99까지 넓혔다
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complaint-model", required=True)
    ap.add_argument("--emotion-model")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "complaint_model_tuning_dev_gold.json"))
    args = ap.parse_args()
    gold = read_gold(GOLD)
    assert all(split_of(int(u.split("#")[0].removeprefix("review:ns"))) == "dev" for u in gold["unit_id"])
    shop, settings, ids = read_shopping(SRC), load_settings(), gold["unit_id"].tolist()

    emotions = {"lexicon": None}
    if args.emotion_model:
        emotions["transformer"] = load_emotion_model(args.emotion_model)
    clf = TransformerComplaintClassifier(args.complaint_model)
    rows = []
    for ename, emotion in emotions.items():
        base = ReviewModels.default()
        if emotion is not None:
            base.emotion = emotion
        r = evaluate_gold(gold, predict_units(ids, shop, settings, models=base))
        rows.append({"emotion": ename, "mode": "rules", "threshold": None, "f1": r["any_complaint"]["f1"],
                     "precision": r["any_complaint"]["precision"], "recall": r["any_complaint"]["recall"], "micro_f1": r["micro"]["f1"]})
        for mode in ("union", "model"):
            for t in args.thresholds:
                clf.mode, clf.threshold = mode, t
                models = ReviewModels.default()
                models.taxonomy = clf
                if emotion is not None:
                    models.emotion = emotion
                r = evaluate_gold(gold, predict_units(ids, shop, settings, models=models))
                rows.append({"emotion": ename, "mode": mode, "threshold": t, "f1": r["any_complaint"]["f1"],
                             "precision": r["any_complaint"]["precision"], "recall": r["any_complaint"]["recall"], "micro_f1": r["micro"]["f1"]})
    for row in rows:
        print(f"{row['emotion']:<11} {row['mode']:<5} {str(row['threshold']):<4} 정밀도 {row['precision']} · 재현율 {row['recall']} · F1 {row['f1']} · 유형 micro F1 {row['micro_f1']}")
    chosen = max(rows, key=lambda r: ((r["f1"] or 0), (r["micro_f1"] or 0)))
    print(f"선택: {chosen}")
    save_result({"gold": str(GOLD.relative_to(REPO_ROOT)), "n": int((~gold["hold"]).sum()), "grid": rows, "chosen": chosen,
                 "rule": "개발용 절 정답에서 불만 여부 F1 최대, 같으면 유형 micro F1 최대"}, args.out)


if __name__ == "__main__":
    main()
