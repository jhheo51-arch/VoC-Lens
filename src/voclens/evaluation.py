"""공개 실제 데이터로 기준선 모델을 평가한다 (docs/evaluation.md).

  kote     KOTE(댓글 5만 문장, 44감정, MIT)로 감정 사전 평가: 극성 정확도, 감정별 정밀도·재현율
  ratings  네이버 쇼핑 리뷰(20만 건, 별점, Public Domain)로 대리 평가: 감성–별점 AUC, 별점별 불만 검출률
  prepare  평가용 절을 무작위로 뽑아 Label Studio 작업 파일 생성 (모델 예측은 보여 주지 않는다)
  gold     직접 라벨링한 평가 세트로 불만 유형별 정밀도·재현율·F1

네이버 쇼핑 리뷰는 줄 번호로 개발용(80%)과 평가용(20%)을 나눈다. 규칙은 개발용만 보고 고치고,
평가용은 라벨링과 최종 측정에만 쓴다.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from voclens.config import load_yaml
from voclens.ingest.review_csv import ReviewRecord
from voclens.pipelines.review import ReviewModels, analyze_records
from voclens.text.emotion import LexiconEmotionModel
from voclens.text.preprocess import preprocess

# KOTE README의 LABELS 순서 (tsv의 라벨 번호가 이 목록의 인덱스)
KOTE_LABELS = [
    "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔", "화남/분노", "존경", "기대감", "우쭐댐/무시함",
    "안타까움/실망", "비장함", "의심/불신", "뿌듯함", "편안/쾌적", "신기함/관심", "아껴주는", "부끄러움", "공포/무서움", "절망",
    "한심함", "역겨움/징그러움", "짜증", "어이없음", "없음", "패배/자기혐오", "귀찮음", "힘듦/지침", "즐거움/신남", "깨달음",
    "죄책감", "증오/혐오", "흐뭇함(귀여움/예쁨)", "당황/난처", "경악", "부담/안_내킴", "서러움", "재미없음", "불쌍함/연민", "놀람",
    "행복", "불안/걱정", "기쁨", "안심/신뢰",
]

# KOTE 44감정 → VoC Lens 7감정. 깨달음·비장함처럼 대응이 애매한 감정은 감정별 평가에서 뺀다.
KOTE_TO_EMOTION = {
    "기쁨": ["환영/호의", "감동/감탄", "고마움", "존경", "기대감", "뿌듯함", "편안/쾌적", "아껴주는", "즐거움/신남",
             "흐뭇함(귀여움/예쁨)", "행복", "기쁨", "안심/신뢰"],
    "슬픔": ["슬픔", "안타까움/실망", "절망", "패배/자기혐오", "서러움", "불쌍함/연민", "힘듦/지침"],
    "분노": ["불평/불만", "화남/분노", "짜증", "지긋지긋", "어이없음"],
    "혐오": ["역겨움/징그러움", "증오/혐오", "한심함", "우쭐댐/무시함"],
    "불안": ["공포/무서움", "불안/걱정", "당황/난처", "부담/안_내킴"],
    "놀람": ["놀람", "경악", "신기함/관심"],
}
_NEGATIVE_EXTRA = {"귀찮음", "재미없음", "의심/불신", "죄책감", "부끄러움"}
KOTE_POSITIVE = set(KOTE_TO_EMOTION["기쁨"])
KOTE_NEGATIVE = {l for e in ("슬픔", "분노", "혐오", "불안") for l in KOTE_TO_EMOTION[e]} | _NEGATIVE_EXTRA

POLARITY_MARGIN = 0.05     # |valence| 가 이보다 작으면 "판단 안 함"으로 본다
TEST_SHARE = 5             # 줄 번호 해시 % 5 == 0 → 평가용 (20%)


# ---------- 공통 지표 ----------

def auc(scores, positive) -> float | None:
    """순위 기반 AUC (Mann–Whitney U / n1·n0). 동점은 0.5로 센다."""
    s = pd.Series(np.asarray(scores, dtype=float))
    y = np.asarray(positive, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = s.rank().to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    # 정답이 있는데 한 번도 맞히지 못하면(재현율 0) 정밀도를 못 구해도 F1은 0으로 센다 (macro 평균이 부풀지 않게)
    f = None if r is None else (2 * p * r / (p + r) if p and r else 0.0)
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f), "support": tp + fn, "predicted": tp + fp}


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


# ---------- KOTE: 감정 사전 평가 ----------

def read_kote(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=["id", "text", "labels"], quoting=3, dtype=str, keep_default_na=False)
    df["labels"] = df["labels"].map(lambda s: [KOTE_LABELS[int(i)] for i in s.split(",") if i.strip()])
    return df


def evaluate_emotions(texts: list[str], gold: list[list[str]], model: LexiconEmotionModel | None = None) -> dict:
    """사전 기반 감정 모델을 KOTE 라벨과 비교한다.

    극성: 긍정 감정만 있거나 부정 감정만 있는 문장에서 Valence 부호가 맞는지 (|V| < 0.05 는 판단 안 함).
    감정별: 감정 표현이 걸린 문장의 대표 감정(중립 제외 최댓값)이 KOTE 라벨 묶음에 있는지.
    """
    model = model or LexiconEmotionModel()
    cleaned = [preprocess(t).text for t in texts]
    results = model.score_many(cleaned) if hasattr(model, "score_many") else [model.score(t) for t in cleaned]
    rows = []
    for res, labels in zip(results, gold):
        non_neutral = {e: p for e, p in res.probs.items() if e != "중립"}
        top = max(non_neutral, key=non_neutral.get) if res.hit_weight > 0 else "중립"
        groups = {e for e, ls in KOTE_TO_EMOTION.items() if set(labels) & set(ls)}
        pos, neg = bool(set(labels) & KOTE_POSITIVE), bool(set(labels) & KOTE_NEGATIVE)
        rows.append({"valence": res.valence, "hit": res.hit_weight > 0, "top": top, "groups": groups,
                     "polarity": "긍정" if pos and not neg else "부정" if neg and not pos else "혼합" if pos else
                     ("없음" if labels == ["없음"] else "기타")})
    df = pd.DataFrame(rows)

    pol = df[df["polarity"].isin(["긍정", "부정"])]
    called = pol[pol["valence"].abs() >= POLARITY_MARGIN]
    correct = ((called["valence"] > 0) == (called["polarity"] == "긍정"))
    per_emotion = {}
    for emotion in KOTE_TO_EMOTION:
        has = df["groups"].map(lambda g, e=emotion: e in g)
        pred = df["top"] == emotion
        per_emotion[emotion] = prf(int((has & pred).sum()), int((~has & pred).sum()), int((has & ~pred).sum()))
    f1s = [m["f1"] for m in per_emotion.values() if m["f1"] is not None]
    none = df[df["polarity"] == "없음"]
    return {
        "n": len(df),
        "coverage": _r(df["hit"].mean()),
        "polarity": {
            "n": len(pol),
            "positive_share": _r((pol["polarity"] == "긍정").mean()),
            "auc": _r(auc(pol["valence"], pol["polarity"] == "긍정")),
            "called_share": _r(len(called) / len(pol)) if len(pol) else None,
            "accuracy_when_called": _r(correct.mean()) if len(called) else None,
            "accuracy_all": _r(correct.sum() / len(pol)) if len(pol) else None,
            "accuracy_negative": _r(correct[called["polarity"] == "부정"].mean()) if len(called) else None,
            "accuracy_positive": _r(correct[called["polarity"] == "긍정"].mean()) if len(called) else None,
        },
        "per_emotion": per_emotion,
        "macro_f1": _r(np.mean(f1s)) if f1s else None,
        "none_false_hit": _r(none["hit"].mean()) if len(none) else None,
    }


# ---------- 네이버 쇼핑 리뷰: 분할·대리 평가 ----------

def split_of(line_no: int) -> str:
    return "test" if zlib.crc32(str(line_no).encode()) % TEST_SHARE == 0 else "dev"


REPEAT_BADGE = "재구매 "   # 네이버 쇼핑이 재구매 고객 리뷰 앞에 붙이는 표시. 고객이 쓴 말이 아니다


def read_shopping(path: str | Path, split: str | None = None) -> pd.DataFrame:
    """naver_shopping.txt (별점\\t리뷰) → line, rating, text, repeat_buyer, split. split을 주면 그 부분만.

    맨 앞의 "재구매 " 표시는 떼어 repeat_buyer 열로 옮긴다 (재구매 의향 신호로 잘못 세지 않게)."""
    df = pd.read_csv(path, sep="\t", names=["rating", "text"], quoting=3, dtype={"rating": int, "text": str}, keep_default_na=False)
    df.insert(0, "line", range(1, len(df) + 1))
    df["repeat_buyer"] = df["text"].str.startswith(REPEAT_BADGE)
    df.loc[df["repeat_buyer"], "text"] = df.loc[df["repeat_buyer"], "text"].str.removeprefix(REPEAT_BADGE)
    df["split"] = df["line"].map(split_of)
    return df[df["split"] == split].reset_index(drop=True) if split else df


def load_emotion_model(path: str | Path):
    """학습한 감정 모델 폴더 → TransformerEmotionModel (pip install -e .[train] 필요)."""
    from voclens.text.emotion_model import TransformerEmotionModel

    return TransformerEmotionModel(path)


def models_with_taxonomy(path: str | Path) -> ReviewModels:
    """분류 기준 YAML 파일을 바꿔 끼운 리뷰 모델 묶음. 이전 버전과 같은 데이터로 비교할 때 쓴다."""
    import yaml

    from voclens.text.journey import JourneyTagger
    from voclens.text.taxonomy import KeywordTaxonomyClassifier, Taxonomy

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tx = Taxonomy.load(data)
    models = ReviewModels.default()
    models.taxonomy_version = str(data.get("version"))
    models.taxonomy = KeywordTaxonomyClassifier(tx)
    models.journey = JourneyTagger(taxonomy=tx)
    return models


def emotion_name(models: ReviewModels | None = None) -> str:
    return (models.emotion if models else LexiconEmotionModel()).name


def taxonomy_version(models: ReviewModels | None = None) -> str:
    return getattr(models, "taxonomy_version", None) or str(load_yaml("taxonomy.yaml").get("version"))


def analyze_shopping(df: pd.DataFrame, settings: dict, domain: str = "product", models: ReviewModels | None = None):
    local = {**settings, "review": {**settings["review"], "domain": domain}}
    records = [ReviewRecord(source_id=f"ns{r.line}", text=r.text, rating=float(r.rating)) for r in df.itertuples()]
    return analyze_records(records, local, models or ReviewModels.default())


def _complaints(unit_labels: list[str], taxonomy) -> list[str]:
    return [l for l in unit_labels if taxonomy.is_complaint(l)]


def evaluate_ratings(df: pd.DataFrame, settings: dict, domain: str = "product", models: ReviewModels | None = None) -> dict:
    """별점을 대리 정답으로 쓴다. 정확도가 아니라 '방향이 맞는지'와 '불만을 얼마나 놓치는지'를 본다.

    - AUC: 리뷰 평균 Valence로 4~5점 리뷰와 1~2점 리뷰를 얼마나 잘 가르는지
    - 별점별 불만 검출률: 1~2점 리뷰 중 불만 유형이 하나라도 붙은 비율(높을수록 덜 놓침),
      5점 리뷰 중 불만이 붙은 비율(낮을수록 좋지만 5점에도 실제 불만이 있을 수 있다)
    """
    models = models or ReviewModels.default()
    tax = models.taxonomy.taxonomy
    sessions = analyze_shopping(df, settings, domain, models)
    rows = []
    for s in sessions:
        vals = [u.valence for u in s.units if u.valence is not None]
        labels = {l for u in s.units for l in _complaints(u.complaint_categories, tax)}
        rows.append({"rating": s.meta["rating"], "valence": float(np.mean(vals)) if vals else 0.0, "labels": labels})
    r = pd.DataFrame(rows)
    r["has"] = r["labels"].map(bool)
    neg = r[r["rating"] <= 2]
    mix = Counter(l for ls in neg["labels"] for l in ls)
    return {
        "n": len(r),
        "taxonomy_version": taxonomy_version(models),
        "emotion_model": emotion_name(models),
        "auc_valence_vs_rating": _r(auc(r["valence"], r["rating"] >= 4)),
        "complaint_rate_by_rating": {int(k): _r(v) for k, v in r.groupby("rating")["has"].mean().items()},
        "negative_review_mix": {k: _r(v / len(neg)) for k, v in mix.most_common()} if len(neg) else {},
    }


# ---------- 평가 세트: 준비와 채점 ----------

def sample_eval_clauses(df: pd.DataFrame, per_rating: int = 50, per_review: int = 2, seed: int = 7) -> pd.DataFrame:
    """평가용(test) 리뷰를 별점마다 같은 수만큼 무작위로 뽑고, 리뷰마다 절을 최대 per_review개 고른다.
    불확실한 절부터 고르는 라벨링(prepare)과 달리 무작위라서 정확도를 편향 없이 잴 수 있다."""
    from voclens.pipelines.review import build_session

    if df.empty:
        return pd.DataFrame(columns=["unit_id", "text", "review", "rating"])
    picked = pd.concat([g.sample(min(per_rating, len(g)), random_state=seed) for _, g in df.groupby("rating")])
    rng = np.random.default_rng(seed)
    rows = []
    for r in picked.itertuples():
        session = build_session(ReviewRecord(source_id=f"ns{r.line}", text=r.text, rating=float(r.rating)))
        units = session.units
        for i in sorted(rng.choice(len(units), size=min(per_review, len(units)), replace=False)) if units else []:
            u = units[int(i)]
            rows.append({"unit_id": u.unit_id, "text": u.text, "review": session.text, "rating": r.rating})
    return pd.DataFrame(rows)


def to_eval_tasks(df: pd.DataFrame) -> list[dict]:
    """labeling/label_studio_config.xml 변수에 맞춘다. 정답이 모델 예측에 끌려가지 않게 예측은 숨긴다."""
    return [{"data": {"unit_id": r.unit_id, "text": r.text, "review": r.review, "zero_shot": "(평가용: 숨김)", "margin": 0.0}}
            for r in df.itertuples()]


def predict_units(unit_ids: list[str], shopping: pd.DataFrame, settings: dict, domain: str = "product",
                  models: ReviewModels | None = None) -> dict[str, list[str]]:
    """평가 세트 절 ID → 현재 규칙의 불만 유형(칭찬·분류 불가 제외)."""
    models = models or ReviewModels.default()
    lines = sorted({int(u.split("#")[0].removeprefix("review:ns")) for u in unit_ids})
    sessions = analyze_shopping(shopping[shopping["line"].isin(lines)], settings, domain, models)
    tax = models.taxonomy.taxonomy
    pred = {u.unit_id: _complaints(u.complaint_categories, tax) for s in sessions for u in s.units}
    return {uid: pred.get(uid, []) for uid in unit_ids}


def read_gold(path: str | Path) -> pd.DataFrame:
    """정답 라벨 읽기. Label Studio 내보내기 JSON 또는 [{"unit_id", "labels", "hold", "memo"}] 목록 둘 다 받는다."""
    from voclens.labeling import read_label_studio_export

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data and ("annotations" in data[0] or "data" in data[0]):
        return read_label_studio_export(path)
    df = pd.DataFrame(data)
    df["hold"] = df.get("hold", False)
    df["hold"] = df["hold"].fillna(False).astype(bool)
    df["memo"] = df.get("memo", "")
    return df[["unit_id", "labels", "hold", "memo"]]


def evaluate_gold(gold: pd.DataFrame, pred: dict[str, list[str]], ignore: tuple[str, ...] = ("칭찬", "분류 불가", "해당 없음")) -> dict:
    """gold: read_label_studio_export 결과(unit_id, labels, hold). 보류한 절은 뺀다.

    불만 유형별 정밀도·재현율·F1, micro/macro 평균, '불만이 있나 없나' 이진 검출 지표를 낸다.
    """
    g = gold[~gold["hold"]]
    truth = {r.unit_id: [l for l in r.labels if l not in ignore] for r in g.itertuples()}
    labels = sorted({l for ls in truth.values() for l in ls} | {l for u in truth for l in pred.get(u, [])})
    per_label, tp_all, fp_all, fn_all = {}, 0, 0, 0
    for label in labels:
        tp = sum(label in truth[u] and label in pred.get(u, []) for u in truth)
        fp = sum(label not in truth[u] and label in pred.get(u, []) for u in truth)
        fn = sum(label in truth[u] and label not in pred.get(u, []) for u in truth)
        per_label[label] = prf(tp, fp, fn)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    has_t = {u: bool(truth[u]) for u in truth}
    has_p = {u: bool(pred.get(u)) for u in truth}
    any_ = prf(sum(has_t[u] and has_p[u] for u in truth), sum(not has_t[u] and has_p[u] for u in truth),
               sum(has_t[u] and not has_p[u] for u in truth))
    f1s = [m["f1"] for m in per_label.values() if m["support"] and m["f1"] is not None]
    return {"n": len(truth), "held": int(gold["hold"].sum()), "any_complaint": any_,
            "micro": prf(tp_all, fp_all, fn_all), "macro_f1": _r(np.mean(f1s)) if f1s else None, "per_label": per_label}


def save_result(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
