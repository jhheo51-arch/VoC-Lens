import pytest

from voicecx.text.segment import split_clauses, split_sentences, segment


@pytest.mark.parametrize(
    "text, expected",
    [
        ("맛있어요. 배달도 빨라요!", ["맛있어요.", "배달도 빨라요!"]),
        ("맛있어요 배달도 빨라요", ["맛있어요", "배달도 빨라요"]),
        ("양이 많음 가격도 착함", ["양이 많음", "가격도 착함"]),
        # 부사 "다"에서 자르면 안 된다
        ("국물까지 다 먹었어요", ["국물까지 다 먹었어요"]),
        ("맛있다고 소문나서 시켰어요", ["맛있다고 소문나서 시켰어요"]),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("맛은 좋은데 배달이 늦어요", ["맛은 좋은데", "배달이 늦어요"]),
        ("양은 많지만 너무 짜요", ["양은 많지만", "너무 짜요"]),
        ("치킨은 맛있어요 근데 콜라가 빠졌어요", ["치킨은 맛있어요", "근데 콜라가 빠졌어요"]),
        ("포장은 깔끔했는데, 다 식었어요", ["포장은 깔끔했는데,", "다 식었어요"]),
        ("가격이 착한데요", ["가격이 착한데요"]),
    ],
)
def test_split_clauses(text, expected):
    assert split_clauses(text) == expected


def test_segment_drops_empty_and_token_only():
    assert segment("맛있어요. [웃음] . ") == ["맛있어요. [웃음]"]
    assert segment("[웃음] . ") == []


@pytest.mark.parametrize(
    "text",
    ["생각보다 맛있어요", "처음 시켜봤어요", "필요 없어요", "배달비 포함 2만원이에요", "다음 주문도 할게요"],
)
def test_split_sentences_noun_exceptions(text):
    assert split_sentences(text) == [text]


def test_split_clauses_generic_de_ending():
    assert split_clauses("가격은 착한데 양이 적어요") == ["가격은 착한데", "양이 적어요"]


def test_segment_keeps_emotion_token_attached():
    assert segment("늦게 왔어요 [슬픔] 다음엔 빨리 와주세요") == ["늦게 왔어요 [슬픔]", "다음엔 빨리 와주세요"]
