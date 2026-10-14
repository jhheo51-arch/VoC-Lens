import pytest

from voclens.text.emotion import LexiconEmotionModel


@pytest.fixture(scope="module")
def model():
    return LexiconEmotionModel()


def top(model, text):
    probs = model.score(text).probs
    return max(probs, key=probs.get)


@pytest.mark.parametrize(
    "text, emotion",
    [
        ("진짜 맛있어요", "기쁨"),
        ("최악이에요 다신 안 시켜요", "분노"),
        ("너무 실망했어요", "슬픔"),
        ("머리카락이 나왔어요 더러워요", "혐오"),
        ("먹고 배탈 날까 봐 걱정돼요", "불안"),
        ("오늘 주문했습니다", "중립"),
    ],
)
def test_top_emotion(model, text, emotion):
    assert top(model, text) == emotion


@pytest.mark.parametrize("text", ["안 좋아요", "좋지 않았어요", "만족스럽지 않아요", "추천 안 해요", "맛있지는 않아요"])
def test_negated_positive_becomes_negative(model, text):
    assert model.score(text).valence < 0


def test_negated_negative_becomes_mildly_positive(model):
    assert model.score("나쁘지 않아요").valence > 0


def test_intensifier_raises_weight(model):
    assert model.score("너무 맛있어요").probs["기쁨"] > model.score("맛있어요").probs["기쁨"]


def test_emotion_token_counts(model):
    assert model.score("늦었네요 [슬픔]").probs["슬픔"] > model.score("늦었네요").probs["슬픔"]


def test_valence_arousal_ranges(model):
    for text in ["맛있어요", "최악", "그냥 그래요", ""]:
        r = model.score(text)
        assert -1 <= r.valence <= 1
        assert 0 <= r.arousal <= 1
        assert abs(sum(r.probs.values()) - 1) < 1e-9


def test_anger_has_higher_arousal_than_sadness(model):
    assert model.score("화나요 짜증나요").arousal > model.score("아쉽고 실망이에요").arousal
