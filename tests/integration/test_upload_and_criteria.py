"""업로드 분석, 판단 근거, 분석 기준 화면 테스트."""

import json

import pandas as pd
import pytest

from voclens.config import REPO_ROOT, load_settings
from voclens.dashboard import criteria, data, upload
from voclens.dashboard.evidence import add_evidence
from voclens.ingest.call_transcript import parse_text_transcript
from voclens.text.cx_signals import CxSignalDetector
from voclens.text.taxonomy import KeywordTaxonomyClassifier

SAMPLES = REPO_ROOT / "data" / "samples"
APP = REPO_ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --- 텍스트 전사본 ---

def test_parse_text_transcript_roles_and_timestamps():
    call = parse_text_transcript("[00:00] 상담원: 고객센터입니다\n[00:05] 고객: 한 시간째 안 와요\n이어지는 말입니다\n[00:12] 상담원: 확인해 드리겠습니다")
    assert [t.role for t in call.turns] == ["agent", "customer", "agent"]
    assert call.turns[1].start == 5 and call.turns[1].end <= 12
    assert call.turns[1].text.endswith("이어지는 말입니다")


def test_parse_text_transcript_estimates_time_and_leaves_ab_roles_to_rules():
    call = parse_text_transcript("A: 무엇을 도와드릴까요\nB: 콜라가 빠졌어요")
    assert all(t.role is None for t in call.turns)
    assert call.turns[1].start > call.turns[0].end


def test_parse_text_transcript_rejects_unformatted():
    with pytest.raises(ValueError):
        parse_text_transcript("그냥 문장만 있습니다")


# --- 업로드 분석 ---

def test_analyze_reviews_upload(settings):
    df = upload.read_table((SAMPLES / "delivery_reviews_synthetic.csv").read_bytes(), "x.csv")
    conn = upload.new_db()
    outcome = upload.analyze_reviews(conn, df.head(60), {"text": "text", "rating": "rating", "created_at": None, "brand": "brand"}, "delivery", settings)
    estimates, evaluation, _ = upload.estimate(conn, settings)
    assert len(outcome.sessions) == 60
    assert data.counts(conn) == {"review": 60}
    assert evaluation is not None                 # 별점 있는 일반 리뷰가 20건 이상
    assert estimates["csat_est"].notna().all()


def test_analyze_reviews_without_rating_notes_and_cp949(settings):
    raw = "내용,기타\n배달이 한 시간 넘게 걸렸어요,a\n맛있어요 또 시킬게요,b\n".encode("cp949")
    df = upload.read_table(raw, "r.csv")
    assert upload.guess_text_column(df) == "내용"
    conn = upload.new_db()
    outcome = upload.analyze_reviews(conn, df, {"text": "내용"}, "delivery", settings)
    estimates, evaluation, _ = upload.estimate(conn, settings)
    assert evaluation is None and estimates["csat_est"].isna().all()
    assert any("CSAT" in n for n in outcome.notes)
    labels = {l for s in outcome.sessions for u in s.units for l in u.complaint_categories}
    assert "배달 지연" in labels


def test_analyze_reviews_row_limit(settings, monkeypatch):
    monkeypatch.setitem(upload.LIMITS, "review_rows", 5)
    df = pd.DataFrame({"text": [f"맛있어요 {i}" for i in range(9)]})
    outcome = upload.analyze_reviews(upload.new_db(), df, {"text": "text"}, "delivery", settings)
    assert len(outcome.sessions) == 5 and "앞의 5건" in outcome.notes[0]


def test_analyze_calls_upload_text_json_and_audio_matching(settings):
    text_call = upload.load_call_upload("my_call.txt", "상담원: 무엇을 도와드릴까요\n고객: 한 시간 넘게 기다렸어요".encode("utf-8"))
    json_call = upload.load_call_upload("c1.json", (SAMPLES / "calls" / "call_001_missing_item.json").read_bytes())
    conn = upload.new_db()
    outcome = upload.analyze_calls(conn, [text_call, json_call], {"unknown": b"x"}, settings, "delivery")
    assert data.counts(conn) == {"call": 2}
    assert any("unknown" in n for n in outcome.notes)
    assert text_call.call_id == "my_call"


def test_uploaded_json_cannot_point_to_server_files(settings, tmp_path):
    raw = json.loads((SAMPLES / "calls" / "call_001_missing_item.json").read_text(encoding="utf-8"))
    raw["audio"] = "C:/Windows/win.ini"
    call = upload.load_call_upload("evil.json", json.dumps(raw).encode("utf-8"))
    assert call.audio is None and call.audio_path() is None

    bundle_raw = json.loads((SAMPLES / "videos" / "video_001_vacuum_review.json").read_text(encoding="utf-8"))
    bundle_raw["audio"] = "/etc/passwd"
    bundle = upload.bundle_from_json(json.dumps(bundle_raw).encode("utf-8"))
    assert bundle.audio is None and bundle.audio_path() is None


def test_analyze_video_from_subtitles(settings):
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n안녕하세요 오늘은 청소기를 소개해요.\n\n00:00:05.000 --> 00:00:12.000\n흡입력이 정말 강력해요. 배터리가 금방 닳아요.\n"
    comments = upload.read_comments("0:06 배터리 공감해요\n좋은 리뷰 감사합니다".encode("utf-8"), "c.txt")
    bundle = upload.bundle_from_subtitles(vtt.encode("utf-8"), comments, {"title": "청소기 리뷰", "domain": "product"})
    conn = upload.new_db()
    outcome = upload.analyze_video(conn, bundle, settings)
    session = outcome.sessions[0]
    assert session.meta["video_type"] == "review"
    assert session.meta["n_timestamp_comments"] == 1
    assert any("조회수" in n for n in outcome.notes)


def test_read_comments_csv_with_likes():
    raw = "댓글,좋아요\n좋아요,3\n별로예요,\n".encode("utf-8")
    comments = upload.read_comments(raw, "c.csv")
    assert [c.likes for c in comments] == [3, 0]


def test_subtitles_without_cues_error():
    with pytest.raises(ValueError):
        upload.bundle_from_subtitles(b"WEBVTT\n\n", [], {})


# --- 판단 근거 ---

def test_evidence_columns_explain_decisions():
    df = pd.DataFrame({"text": ["진짜 맛있어요", "콜라가 빠져 있었어요 다신 안 시켜요", "추천 안 해요"], "valence": [0.5, -0.4, -0.2]})
    out = add_evidence(df, "delivery")
    assert "맛있(기쁨, 강조)" in out.loc[0, "감정 근거"]
    assert "누락·오배송 ← '빠져 있'(불만 표현)" in out.loc[1, "불만 근거"]
    assert "이탈 의사" in out.loc[1, "신호 근거"]
    assert "부정 → 슬픔" in out.loc[2, "감정 근거"]


def test_taxonomy_explain_topic_needs_negative_valence():
    clf = KeywordTaxonomyClassifier()
    assert clf.explain("가격이 좀 있어요", 0.3) == ""
    assert "주제 + 부정 감성" in clf.explain("가격이 좀 있어요", -0.3)


def test_cx_explain_respects_exclude():
    assert CxSignalDetector().explain("재주문 안 할 거예요").startswith("이탈 의사")


# --- 분석 기준 ---

def test_criteria_tables_come_from_configs():
    assert len(criteria.taxonomy_table()) == 21
    assert set(criteria.emotion_table()["감정"]) == {"기쁨", "슬픔", "분노", "불안", "놀람", "혐오", "중립"}
    assert len(criteria.signal_table()) >= 8
    assert "0.4" in " ".join(criteria.call_rules())        # repeat_similarity 설정값이 문장에 들어간다
    assert len(criteria.video_tables()["types"]) == 5
    assert all(isinstance(r, str) and r for r in criteria.review_rules() + criteria.cx_rules() + criteria.video_rules())


# --- 화면 ---

def _app(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.delenv("VOCLENS_DASHBOARD_ADMIN", raising=False)
    at = testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_criteria_page(monkeypatch):
    at = _app(monkeypatch)
    at.sidebar.radio(key="page").set_value("분석 기준").run()
    assert not at.exception
    assert len(at.dataframe) >= 6


def test_upload_page_samples_run(monkeypatch):
    at = _app(monkeypatch)
    at.sidebar.radio(key="page").set_value("직접 분석").run()
    assert not at.exception
    at.button(key="up_review_sample").click().run()
    at.button(key="up_review_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "up_review" in at.session_state
    at.button(key="up_call_sample").click().run()
    at.button(key="up_call_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "up_call" in at.session_state
    at.button(key="up_video_sample").click().run()
    at.button(key="up_video_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "up_video" in at.session_state
