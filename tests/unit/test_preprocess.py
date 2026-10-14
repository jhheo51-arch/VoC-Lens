import pytest

from voclens.text.preprocess import (
    detect_review_event,
    mask_pii,
    normalize_repeats,
    preprocess,
    replace_emotion_cues,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("맛있어요ㅋㅋㅋㅋㅋㅋ", "맛있어요ㅋㅋ"),
        ("최악!!!!!!", "최악!!"),
        ("너무너무너무 좋아요", "너무너무 좋아요"),
        ("배고파요......", "배고파요.."),
        ("평범한 문장", "평범한 문장"),
        ("10000원 쿠폰", "10000원 쿠폰"),
    ],
)
def test_normalize_repeats(raw, expected):
    assert normalize_repeats(raw) == expected


@pytest.mark.parametrize(
    "raw, token",
    [
        ("늦게 와서 슬퍼요ㅠㅠ", "[슬픔]"),
        ("ㅜㅜ 아쉽네요", "[슬픔]"),
        ("맛있어요ㅋㅋ", "[웃음]"),
        ("최고예요 😍", "[기쁨]"),
        ("화나네요 😡", "[분노]"),
        ("눈물나요 😭", "[슬픔]"),
        ("굿 👍", "[기쁨]"),
    ],
)
def test_replace_emotion_cues(raw, token):
    assert token in replace_emotion_cues(raw)


def test_emotion_cues_keep_text():
    assert replace_emotion_cues("맛있어요ㅠㅠ").startswith("맛있어요")


@pytest.mark.parametrize(
    "raw, masked",
    [
        ("010-1234-5678로 연락주세요", "[전화번호]로 연락주세요"),
        ("01012345678 연락 부탁", "[전화번호] 연락 부탁"),
        ("메일은 abc.def@example.com 입니다", "메일은 [이메일] 입니다"),
        ("주문번호 2026101412345 확인", "주문번호 [번호] 확인"),
    ],
)
def test_mask_pii(raw, masked):
    assert mask_pii(raw) == masked


def test_mask_pii_keeps_short_numbers():
    assert mask_pii("1시간 걸렸고 2만원이에요") == "1시간 걸렸고 2만원이에요"


@pytest.mark.parametrize(
    "text",
    [
        "리뷰이벤트로 콜라 받았어요",
        "리뷰 이벤트 참여합니다",
        "리뷰 약속하고 서비스 받았어요",
        "서비스 감사합니다 잘 먹었어요",
        "감자튀김 서비스 감사해요",
    ],
)
def test_detect_review_event_true(text):
    assert detect_review_event(text)


@pytest.mark.parametrize("text", ["배달이 빨라요", "서비스가 별로예요", "고객 서비스 최악"])
def test_detect_review_event_false(text):
    assert not detect_review_event(text)


def test_preprocess_returns_flags():
    result = preprocess("리뷰이벤트 참여!! 맛있어요ㅋㅋㅋㅋ 010-1111-2222")
    assert "review_event" in result.flags
    assert "pii_masked" in result.flags
    assert "[전화번호]" in result.text
    assert "ㅋㅋㅋ" not in result.text


def test_preprocess_collapses_whitespace():
    assert preprocess("  맛있어요   \n\n 또 시킬게요  ").text == "맛있어요 또 시킬게요"
