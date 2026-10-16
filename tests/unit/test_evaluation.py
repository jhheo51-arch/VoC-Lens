"""공개 데이터 평가 도구 테스트: 지표 계산, 분할, KOTE 매핑, 평가 세트 채점. 외부 데이터 없이 작은 표본으로."""

import pandas as pd
import pytest

from voclens import evaluation as ev
from voclens.config import load_settings
from voclens.text.taxonomy import KeywordTaxonomyClassifier


def test_auc_and_prf():
    assert ev.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert ev.auc([0.5, 0.5], [True, False]) == 0.5               # 동점은 절반
    assert ev.auc([1, 2], [True, True]) is None
    m = ev.prf(tp=8, fp=2, fn=8)
    assert (m["precision"], m["recall"], m["f1"], m["support"]) == (0.8, 0.5, 0.615, 16)


def test_kote_mapping_covers_known_labels():
    mapped = {l for ls in ev.KOTE_TO_EMOTION.values() for l in ls}
    assert mapped <= set(ev.KOTE_LABELS) and len(ev.KOTE_LABELS) == 44
    assert not (ev.KOTE_POSITIVE & ev.KOTE_NEGATIVE)


def test_read_kote_turns_indices_into_names(tmp_path):
    path = tmp_path / "k.tsv"
    path.write_text("1\t너무 고마워요\t4,42\n2\t짜증나네\t0,22,\n", encoding="utf-8")
    df = ev.read_kote(path)
    assert df["labels"].tolist() == [["고마움", "기쁨"], ["불평/불만", "짜증"]]


def test_evaluate_emotions_polarity():
    texts = ["정말 맛있고 좋아요 최고", "너무 화나고 짜증나요 최악", "그냥 그래요"]
    gold = [["기쁨", "행복"], ["화남/분노", "짜증"], ["없음"]]
    result = ev.evaluate_emotions(texts, gold)
    assert result["polarity"]["n"] == 2
    assert result["polarity"]["accuracy_when_called"] == 1.0
    assert result["per_emotion"]["기쁨"]["recall"] == 1.0


def test_split_is_deterministic_and_about_one_fifth():
    splits = [ev.split_of(i) for i in range(1, 5001)]
    assert splits == [ev.split_of(i) for i in range(1, 5001)]
    assert 0.17 < splits.count("test") / 5000 < 0.23


def _shopping(tmp_path):
    rows = ["5\t배송 빠르고 좋아요", "1\t블루를 샀는데 블랙이 왔어요. 너무 실망입니다",
            "2\t한 번 빨았더니 실밥이 다 풀렸어요", "4\t괜찮아요 다만 냄새가 좀 심해요"] * 5
    path = tmp_path / "naver_shopping.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_read_shopping_and_ratings_proxy(tmp_path):
    df = ev.read_shopping(_shopping(tmp_path))
    assert df["line"].tolist()[:2] == [1, 2] and set(df["split"]) <= {"dev", "test"}
    result = ev.evaluate_ratings(df, load_settings())
    assert result["complaint_rate_by_rating"][1] == 1.0 and result["complaint_rate_by_rating"][5] == 0.0
    assert "오배송·누락" in result["negative_review_mix"] and "품질·마감" in result["negative_review_mix"]
    assert result["auc_valence_vs_rating"] > 0.5


def test_eval_sample_hides_predictions_and_gold_scoring(tmp_path):
    df = ev.read_shopping(_shopping(tmp_path))
    sample = ev.sample_eval_clauses(df, per_rating=2)
    assert sample.groupby("rating").size().max() <= 4                  # 별점별 2건 × 리뷰당 절 2개
    tasks = ev.to_eval_tasks(sample)
    assert all(t["data"]["zero_shot"] == "(평가용: 숨김)" for t in tasks)

    pred = ev.predict_units(sample["unit_id"].tolist(), df, load_settings())
    assert set(pred) == set(sample["unit_id"])
    gold = pd.DataFrame({
        "unit_id": ["a", "b", "c", "d"],
        "labels": [["오배송·누락"], ["품질·마감"], ["칭찬"], ["냄새"]],
        "hold": [False, False, False, True],
    })
    result = ev.evaluate_gold(gold, {"a": ["오배송·누락"], "b": [], "c": ["가격"], "d": ["냄새"]})
    assert result["n"] == 3 and result["held"] == 1                     # 보류는 뺀다
    assert result["per_label"]["오배송·누락"]["f1"] == 1.0
    assert result["per_label"]["품질·마감"]["recall"] == 0.0
    assert result["any_complaint"]["precision"] == 0.5                  # 칭찬 절에 가격을 붙인 오탐


@pytest.mark.parametrize("text, expected", [
    ("배송 빠르고 불량이 없습니다", set()),
    ("벗겨지지 않고 잘 신겨요", set()),
    ("상한 거 없이 잘 왔어요", set()),
    ("아삭아삭 맛있어요", set()),
    ("어머니가 맨날 끌어안고 주무시네요", set()),
    ("불량이에요 바로 반품했어요", {"제품 불량"}),
    ("테두리가 다 삭아 있어요", {"품질·마감"}),
    ("블루를 샀는데 블랙이 왔어요", {"오배송·누락"}),
    ("포장이 다 터져서 왔습니다", {"포장·파손"}),
    ("배송이 일주일이나 걸렸어요", {"배송 지연"}),
])
def test_product_rules_and_negation(text, expected):
    hits = {label for label, strength in KeywordTaxonomyClassifier().classify(text, 0.3, "product") if strength >= 1.0}
    assert hits == expected


def test_cross_clause_complaint_goes_to_later_clause():
    from voclens.schema import SourceType, Unit

    units = [Unit(unit_id=f"s#{i}", session_id="s", source_type=SourceType.REVIEW, order=i, text=t, clean_text=t, valence=0.0)
             for i, t in enumerate(["블루를 샀는데", "블랙이 왔어요.", "다음엔 잘 보내 주세요"])]
    KeywordTaxonomyClassifier().predict(units, "product")
    assert "오배송·누락" not in units[0].complaint_categories
    assert "오배송·누락" in units[1].complaint_categories
    assert "오배송·누락" not in units[2].complaint_categories          # 앞 절 안에서 끝난 매치는 다시 세지 않는다


def test_cli_eval_ratings_and_prepare(tmp_path, capsys):
    import json

    from voclens.cli import main

    src = _shopping(tmp_path)
    out = tmp_path / "r.json"
    assert main(["eval", "ratings", str(src), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 20
    tasks = tmp_path / "tasks.json"
    assert main(["eval", "prepare", str(src), "--per-rating", "1", "--out", str(tasks)]) == 0
    assert "원문이 들어 있으니" in capsys.readouterr().out
    # prepare는 평가용(test) 분할만 쓴다. 20줄 중 평가용이 없으면 빈 목록이어도 된다
    assert isinstance(json.loads(tasks.read_text(encoding="utf-8")), list)


def test_read_gold_plain_list_and_old_taxonomy(tmp_path):
    import json

    import yaml

    from voclens.config import load_yaml

    path = tmp_path / "gold.json"
    path.write_text(json.dumps([{"unit_id": "u1", "labels": ["냄새"], "hold": False, "memo": ""},
                                {"unit_id": "u2", "labels": ["칭찬"]}], ensure_ascii=False), encoding="utf-8")
    gold = ev.read_gold(path)
    assert gold["hold"].tolist() == [False, False] and gold["labels"].tolist()[0] == ["냄새"]

    old = load_yaml("taxonomy.yaml")
    old["version"] = "0.0-test"
    old["subcategories"] = {k: v for k, v in old["subcategories"].items() if k != "냄새"}
    tpath = tmp_path / "old.yaml"
    tpath.write_text(yaml.safe_dump(old, allow_unicode=True), encoding="utf-8")
    models = ev.models_with_taxonomy(tpath)
    assert ev.taxonomy_version(models) == "0.0-test"
    result = ev.evaluate_ratings(ev.read_shopping(_shopping(tmp_path)), load_settings(), models=models)
    assert result["taxonomy_version"] == "0.0-test" and "냄새" not in result["negative_review_mix"]


def test_never_predicted_label_counts_as_zero_f1():
    gold = pd.DataFrame({"unit_id": ["a", "b"], "labels": [["성능"], ["냄새"]], "hold": [False, False]})
    result = ev.evaluate_gold(gold, {"a": [], "b": ["냄새"]})
    assert result["per_label"]["성능"]["f1"] == 0.0
    assert result["macro_f1"] == 0.5


def test_transformer_result_maps_group_probs_to_valence():
    """학습 모델의 6감정 확률 → 7감정·Valence 변환 (torch 없이 변환 로직만)."""
    import numpy as np

    from voclens.config import load_yaml
    from voclens.text.emotion_model import TransformerEmotionModel

    m = object.__new__(TransformerEmotionModel)
    m.circumplex = {e: tuple(v) for e, v in load_yaml("lexicons/emotion.yaml")["circumplex"].items()}
    joy = m._result(np.array([0.95, 0.02, 0.02, 0.01, 0.01, 0.05]))
    anger = m._result(np.array([0.03, 0.30, 0.92, 0.40, 0.05, 0.05]))
    flat = m._result(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
    assert joy.valence > 0.25 and anger.valence < -0.2               # 분류 기준의 칭찬·분류 불가 임계치와 맞물린다
    assert flat.hit_weight == 0.0 and abs(flat.valence) < 0.1 and flat.probs["중립"] > 0.5
    assert abs(sum(joy.probs.values()) - 1) < 1e-9


def test_complaint_model_union_and_model_modes():
    """학습 불만 분류기의 합치기 로직 (모델 확률은 가짜로 넣는다)."""
    import numpy as np

    from voclens.schema import SourceType, Unit
    from voclens.text.complaint_model import TransformerComplaintClassifier, model_input
    from voclens.text.taxonomy import KeywordTaxonomyClassifier

    def make(mode):
        clf = object.__new__(TransformerComplaintClassifier)
        KeywordTaxonomyClassifier.__init__(clf)
        clf.labels, clf.threshold, clf.mode = ["사이즈·색상", "제품 불량"], 0.5, mode
        clf._cache = {"s#0": np.array([0.9, 0.1]), "s#1": np.array([0.1, 0.2])}
        clf.prepare = lambda units: None
        return clf

    def units():
        return [Unit(unit_id=f"s#{i}", session_id="s", source_type=SourceType.REVIEW, order=i, text=t, clean_text=t, valence=v)
                for i, (t, v) in enumerate([("바지가 생각보다 껴요", 0.1), ("불량이에요", 0.0)])]

    u = make("union").predict(units(), "product")
    assert u[0].complaint_categories == ["사이즈·색상"] and u[0].valence <= -0.3      # 규칙이 놓친 불만을 보충
    assert u[1].complaint_categories == ["제품 불량"]                                   # 규칙이 잡은 불만은 유지
    m = make("model").predict(units(), "product")
    assert m[1].complaint_categories == [] and m[1].valence == 0.0                     # 모델만: 규칙 결과를 버리고 Valence 복원
    assert model_input("블랙이 왔어요", "블루를 샀는데") == "블루를 샀는데 [SEP] 블랙이 왔어요"
