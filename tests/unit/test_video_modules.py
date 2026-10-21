import pytest

from voclens.config import load_yaml
from voclens.ingest.subtitles import Cue, cues_to_sentences, dedupe_rolling, parse_subtitles, parse_timestamp
from voclens.ingest.youtube import parse_comment_threads, video_id_from_url
from voclens.pipelines.video import audience_engagement
from voclens.text.emotion import LexiconEmotionModel
from voclens.text.taxonomy import KeywordTaxonomyClassifier, normalize_domain
from voclens.video.classify import VideoTypeClassifier
from voclens.video.comments import compare_creator_audience, extract_timestamps, find_segment
from voclens.video.segment import TextTilingSegmenter

VTT = """WEBVTT
Kind: captions
Language: ko

00:00:01.000 --> 00:00:03.500 align:start position:0%
안녕하세요 오늘은 청소기를

00:00:03.500 --> 00:00:06.000
<c>소개해 드릴게요.</c> 흡입력이

00:00:06.000 --> 00:00:09.000
정말 강력해요.
"""

SRT = """1
00:00:01,000 --> 00:00:02,500
첫 번째 자막입니다.

2
00:00:02,500 --> 00:00:04,000
두 번째 자막입니다.
"""


@pytest.mark.parametrize(
    "value, seconds",
    [("00:00:01.000", 1.0), ("01:02.5", 62.5), ("1:00:00,250", 3600.25), ("00:00:09.12", 9.12)],
)
def test_parse_timestamp(value, seconds):
    assert parse_timestamp(value) == pytest.approx(seconds)


def test_parse_vtt_strips_tags_and_settings():
    cues = parse_subtitles(VTT)
    assert [c.text for c in cues] == ["안녕하세요 오늘은 청소기를", "소개해 드릴게요. 흡입력이", "정말 강력해요."]
    assert cues[0].start == 1.0 and cues[-1].end == 9.0


def test_parse_srt():
    cues = parse_subtitles(SRT)
    assert len(cues) == 2 and cues[1].start == 2.5


def test_dedupe_rolling_captions():
    cues = [Cue(0, 2, "안녕하세요"), Cue(2, 4, "안녕하세요 반갑습니다"), Cue(4, 5, "안녕하세요 반갑습니다")]
    out = dedupe_rolling(cues)
    assert [c.text for c in out] == ["안녕하세요", "반갑습니다"]
    assert out[-1].end == 5


def test_cues_to_sentences_aligns_times():
    sentences = cues_to_sentences(parse_subtitles(VTT))
    texts = [s.text for s in sentences]
    assert texts[-1] == "흡입력이 정말 강력해요."
    last = sentences[-1]
    assert last.start == 3.5 and last.end == 9.0   # 두 큐에 걸친 문장


def test_cues_to_sentences_breaks_at_long_gap():
    # 마침표 없는 자막에서 7분 떨어진 큐가 한 문장으로 붙지 않아야 한다 (실제 영상에서 발견)
    cues = [Cue(990, 993, "배터리는 생각보다 오래 가고"), Cue(1406, 1408, "카메라는 아쉽네요")]
    sentences = cues_to_sentences(cues)
    assert [(s.start, s.end) for s in sentences] == [(990, 993), (1406, 1408)]


@pytest.mark.parametrize(
    "url, vid",
    [
        ("https://www.youtube.com/watch?v=abcdefghijk&t=10s", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk", "abcdefghijk"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
    ],
)
def test_video_id_from_url(url, vid):
    assert video_id_from_url(url) == vid


def test_video_id_invalid():
    with pytest.raises(ValueError):
        video_id_from_url("https://example.com/video")


def test_parse_comment_threads():
    payload = {"items": [
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "1:10 좋아요", "likeCount": 3, "publishedAt": "2026-10-01T00:00:00Z"}}}},
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "  "}}}},
    ]}
    comments = parse_comment_threads(payload)
    assert len(comments) == 1 and comments[0].likes == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3:20 여기 공감", [200.0]),
        ("1:02:03 부분", [3723.0]),
        ("0:40이랑 2:00 둘 다", [40.0, 120.0]),
        ("16:9 화면비", []),          # 초가 두 자리가 아니라 시각이 아님
        ("12:75는 시각이 아님", []),
        ("2026:10:14", []),
    ],
)
def test_extract_timestamps(text, expected):
    assert extract_timestamps(text) == expected


def test_extract_timestamps_respects_duration():
    # "16:10" 화면비는 형식상 시각과 구분할 수 없어서 영상 길이로 거른다
    assert extract_timestamps("16:10 화면비") == [970.0]
    assert extract_timestamps("16:10 화면비", max_sec=300) == []


def test_find_segment_with_tolerance():
    spans = [(0, 10), (12, 20)]
    assert find_segment(5, spans) == 0
    assert find_segment(11.5, spans, tolerance=2) == 1   # 12초 구간이 더 가깝다
    assert find_segment(40, spans, tolerance=5) is None


def test_compare_creator_audience():
    result = compare_creator_audience([0.1, 0.3], [0.5])
    assert result["audience_minus_creator"] == pytest.approx(0.3)
    assert compare_creator_audience([0.2], [])["audience_minus_creator"] is None


def test_video_type_classifier():
    clf = VideoTypeClassifier()
    review = clf.classify({"title": "무선청소기 솔직 리뷰", "tags": ["내돈내산"]}, [])
    vlog = clf.classify({"title": "직장인 브이로그 하루 일상"}, [])
    other = clf.classify({"title": "제목"}, ["내용"])
    assert review.type == "review" and review.confidence > 0.5
    assert vlog.type == "vlog"
    assert other.type == "other" and other.confidence == 0


def test_segmenter_splits_topic_shift_and_labels_verdict():
    sentences = [
        "안녕하세요 오늘은 청소기를 소개해요.", "구독 부탁드려요.", "오늘 영상 시작할게요.",
        "박스를 개봉해 볼게요.", "구성품은 본체와 헤드예요.", "포장이 깔끔해요.",
        "이제 써 볼게요.", "흡입력이 강력해요.", "소음은 조금 있어요.",
        "총평을 하자면 추천해요.", "가격은 조금 비싸요.", "저는 만족해요.",
    ]
    segments = TextTilingSegmenter().segment(sentences)
    assert len(segments) >= 3
    assert segments[0].label == "도입"
    assert segments[-1].label == "총평"
    assert segments[0].start == 0 and segments[-1].end == len(sentences)
    assert all(b.start == a.end for a, b in zip(segments, segments[1:]))


def test_segmenter_short_input_is_one_segment():
    assert len(TextTilingSegmenter().segment(["하나.", "둘."])) == 1


def test_audience_engagement():
    assert audience_engagement({}, 10, 2) is None
    high = audience_engagement({"views": 1000, "likes": 50}, 5, 5)
    low = audience_engagement({"views": 100000, "likes": 100}, 5, 0)
    assert high == 100.0 and low < high


@pytest.mark.parametrize(
    "value, domain", [("배달", "delivery"), ("Product", "product"), ("가전", "product"), (None, None), ("기타", None)]
)
def test_normalize_domain(value, domain):
    assert normalize_domain(value) == domain


def test_taxonomy_domain_filter():
    clf = KeywordTaxonomyClassifier()
    text = "충전이 오래 걸리는 것도 아쉬워요"
    assert "배달 지연" in [l for l, _ in clf.classify(text, -0.3)]
    product = [l for l, _ in clf.classify(text, -0.3, domain="product")]
    assert "배달 지연" not in product and "배터리" in product


def test_defect_keyword_does_not_match_haja_myeon():
    assert KeywordTaxonomyClassifier().classify("총평을 하자면 맛은 추천해요", 0.3) == []


@pytest.mark.parametrize("text", ["배달만 빨랐으면 재주문할 것 같아요", "배터리는 더 좋아야 한다고 생각해요"])
def test_conditional_positive_is_not_praise(text):
    assert LexiconEmotionModel().score(text).valence < 0


def test_conditional_does_not_flip_negative_emotion():
    assert LexiconEmotionModel().score("아이가 자고 있으면 깜짝 놀라요").valence < 0.2


def _position_sentences():
    # 0~3 도입, 4~7 사용, 8~11 또 '오늘은'(중간 도입 키워드), 12~15 사용, 16~19 총평
    return [
        "안녕하세요 오늘은 청소기를 소개해요.", "오늘 준비했습니다.", "오늘은 이걸 볼게요.", "시작합니다.",
        "이제 써 볼게요.", "흡입력 테스트를 해요.", "소리가 조금 커요.", "모드를 바꿔 써보면 달라요.",
        "오늘은 날씨가 좋네요.", "안녕하세요 다시 왔어요.", "오늘은 정말 덥네요.", "준비했던 걸 꺼냈어요.",
        "다시 써 볼게요.", "화면 모드를 테스트해요.", "소리는 괜찮아요.", "쓰면 편해요.",
        "총평을 하자면 추천해요.", "결론은 살 만해요.", "정리하면 만족해요.", "별점은 4점이에요.",
    ]


def test_position_segmenter_keeps_intro_front_and_verdict_back():
    from voclens.video.segment import PositionalSegmenter

    sentences = _position_sentences()
    segments = PositionalSegmenter().segment(sentences)
    labels = [s.label for s in segments]
    n = len(sentences)
    assert labels[0] == "도입" and labels[-1] == "총평"
    assert all(s.start / n <= 0.2 for s in segments if s.label == "도입")
    assert all((s.start + s.end) / 2 / n >= 0.7 for s in segments if s.label == "총평")
    assert all(a != b for a, b in zip(labels, labels[1:]))          # 같은 이름 이웃은 합친다
    assert segments[0].start == 0 and segments[-1].end == n
    assert all(b.start == a.end for a, b in zip(segments, segments[1:]))


def test_position_segmenter_uses_time_positions():
    from voclens.video.segment import PositionalSegmenter

    sentences = _position_sentences()
    # 마지막 네 문장이 영상 중간(0.4~0.5)에 있다고 하면 총평으로 보지 않는다
    positions = [i / 40 for i in range(len(sentences))]
    labels = [s.label for s in PositionalSegmenter().segment(sentences, positions=positions)]
    assert "총평" not in labels


def test_position_segmenter_suppresses_label_by_product_group():
    from voclens.video.segment import PositionalSegmenter, product_group

    cfg = load_yaml("video.yaml")
    groups = cfg["position_segmentation"]["product_groups"]
    assert product_group({"title": "신형 세단 시승기"}, groups) == "자동차"
    assert product_group({"title": "무선 청소기 리뷰"}, groups) is None
    sentences = [
        "안녕하세요 오늘은 차를 소개해요.", "준비했습니다.", "시작할게요.",
        "박스 개봉해 볼게요.", "구성품이 많아요.", "포장도 봐요.",
        "주행해 볼게요.", "소리가 조용해요.", "모드를 바꿔요.",
        "총평은 추천이에요.", "결론은 살 만해요.", "정리하면 만족이에요.",
    ]
    car = [s.label for s in PositionalSegmenter().segment(sentences, meta={"title": "신형 세단 시승기"})]
    other = [s.label for s in PositionalSegmenter().segment(sentences, meta={"title": "청소기 리뷰"})]
    assert "언박싱" not in car
    assert "언박싱" in other


def test_make_segmenter_default_is_texttiling():
    from voclens.video.segment import PositionalSegmenter, TextTilingSegmenter, make_segmenter

    assert type(make_segmenter()) is TextTilingSegmenter
    assert isinstance(make_segmenter("position"), PositionalSegmenter)
    with pytest.raises(ValueError):
        make_segmenter("embedding")
