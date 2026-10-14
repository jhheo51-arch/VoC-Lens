import numpy as np
import pytest

from voclens.schema import SourceType, Unit
from voclens.text.cx_signals import CxSignalDetector
from voclens.text.journey import JourneyTagger
from voclens.text.taxonomy import (
    COMPLAINT_VALENCE_CAP,
    EmbeddingTaxonomyClassifier,
    KeywordTaxonomyClassifier,
    Taxonomy,
    uncertainty_ranking,
)


def make_unit(text, valence=0.0):
    return Unit(unit_id="u", session_id="s", source_type=SourceType.REVIEW, order=0, text=text, clean_text=text, valence=valence)


@pytest.fixture(scope="module")
def clf():
    return KeywordTaxonomyClassifier()


@pytest.mark.parametrize(
    "text, label",
    [
        ("한 시간 넘게 걸려서 왔어요", "배달 지연"),
        ("1시간 만에 왔네요", "배달 지연"),
        ("콜라가 빠져 있었어요", "누락·오배송"),
        ("다 식어서 왔어요", "온도·식감"),
        ("너무 짜서 못 먹었어요", "맛"),
        ("머리카락이 나왔어요", "위생·이물질"),
        ("국물이 다 새서 왔어요", "포장 상태"),
        ("이 가격이면 너무 비싸요", "가격"),
        ("가게에 전화해도 전화를 안 받아요", "문의·연락"),
        ("오이 빼달라고 했는데 들어 있어요", "요청 미반영"),
        ("결제가 안 돼서 세 번 시도했어요", "앱·결제 오류"),
    ],
)
def test_keyword_complaint(clf, text, label):
    labels = [l for l, _ in clf.classify(text, valence=0.0)]
    assert label in labels


def test_topic_requires_negative_valence(clf):
    assert clf.classify("요청사항 잘 들어주셨어요", valence=0.5) == []
    assert "요청 미반영" in [l for l, _ in clf.classify("요청사항 적었는데 그대로네요", valence=-0.4)]


def test_predict_praise_and_unknown(clf):
    praise = clf.predict([make_unit("맛있어요", valence=0.5)])[0]
    unknown = clf.predict([make_unit("그냥 그랬어요", valence=-0.3)])[0]
    neutral = clf.predict([make_unit("주문했습니다", valence=0.0)])[0]
    assert praise.complaint_categories == ["칭찬"]
    assert unknown.complaint_categories == ["분류 불가"]
    assert neutral.complaint_categories == []


def test_predict_caps_valence_for_explicit_complaint(clf):
    unit = clf.predict([make_unit("한 시간 넘게 걸렸어요", valence=0.0)])[0]
    assert unit.valence == COMPLAINT_VALENCE_CAP
    assert unit.aspect == "배달"
    assert unit.aspect_sentiment == COMPLAINT_VALENCE_CAP


def test_predict_finds_aspect_for_praise(clf):
    unit = clf.predict([make_unit("배달이 빨리 왔어요", valence=0.4)])[0]
    assert unit.aspect == "배달"
    assert unit.aspect_sentiment == 0.4


def test_taxonomy_lookup():
    tx = Taxonomy.load()
    assert tx.category_of("배달 지연") == "배송·이행"
    assert tx.category_of("칭찬") == "칭찬"
    assert tx.difficulty_of("누락·오배송") == 1
    assert not tx.is_complaint("칭찬")


def _char_bag_encoder(texts):
    """테스트용 가짜 인코더: 글자 빈도 벡터 (한 프로세스 안에서는 hash가 일정하다)."""
    out = np.zeros((len(texts), 4096))
    for r, t in enumerate(texts):
        for ch in t:
            out[r, hash(ch) % 4096] += 1
    return out


def test_embedding_classifier_with_fake_encoder():
    clf = EmbeddingTaxonomyClassifier(encode=_char_bag_encoder, threshold=0.0)
    results = clf.classify(["한 시간 넘게 걸려서 왔어요", "콜라가 누락됐어요"])
    assert results[0][0] == "배달 지연"
    assert results[1][0] == "누락·오배송"
    assert all(-1 <= sim <= 1 for _, sim, _ in results)


def test_uncertainty_ranking_orders_by_margin():
    results = [("a", 0.9, 0.5), ("b", 0.6, 0.01), (None, 0.3, 0.1)]
    assert uncertainty_ranking(results) == [1, 2, 0]


@pytest.mark.parametrize(
    "text, stage",
    [("배달이 빨리 왔어요", "배달"), ("재주문할게요", "재주문"), ("결제할 때 쿠폰이 안 먹혀요", "주문·결제")],
)
def test_journey_keywords(text, stage):
    assert JourneyTagger().tag(text, []) == stage


def test_journey_uses_taxonomy_stage_first():
    assert JourneyTagger().tag("맛있는데 늦었어요", ["배달 지연"]) == "배달"


@pytest.mark.parametrize(
    "text, signal, expected",
    [
        ("또 시킬게요", "repurchase_intent", True),
        ("재주문 안 할 거예요", "repurchase_intent", False),
        ("재주문 안 할 거예요", "churn_intent", True),
        ("다신 안 시켜요", "churn_intent", True),
        ("강추합니다", "recommend_intent", True),
        ("비추예요", "recommend_intent", False),
        ("비추예요", "not_recommend", True),
        ("세 번이나 전화했어요", "repeat_contact", True),
        ("한 시간 기다렸어요", "waiting", True),
        ("사장님이 바로 다시 보내주셨어요", "resolution", True),
        ("사장님이 사과 전화 주셨어요", "apology_received", True),
    ],
)
def test_cx_signals(text, signal, expected):
    assert CxSignalDetector().detect(text).get(signal, False) is expected
