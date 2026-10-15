"""대시보드 테스트: 데이터 조회 함수, 차트 사양, Streamlit 화면 실행."""

import pandas as pd
import pytest

from voclens.config import REPO_ROOT, load_settings
from voclens.cx import compare
from voclens.cx.cx_report import run_cx
from voclens.dashboard import charts, data
from voclens.store import db

APP = REPO_ROOT / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def sample_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("dash") / "dash.db"
    data.build_sample_db(path, load_settings())
    return path


@pytest.fixture
def conn(sample_db):
    c = db.connect(sample_db)
    yield c
    c.close()


def test_build_sample_db_fills_all_channels(conn):
    assert data.counts(conn) == {"review": 180, "call": 4, "video": 2}
    assert data.has_estimates(conn)
    assert data.brands(conn) == ["치킨브랜드A", "피자브랜드B"]


def test_call_list_and_detail(conn):
    calls = data.call_list(conn)
    assert calls.iloc[0]["call_id"] == "call_002_late_delivery"     # 좌절 높은 순
    assert set(calls["outcome"]) == {"해결", "보류", "이관"}
    units = data.session_units(conn, "call:call_001_missing_item")
    assert {"고객", "상담원"} == set(units["role"])
    assert {"stress", "frustration", "complaints", "signals"} <= set(units.columns)
    events = data.session_events(conn, "call:call_001_missing_item")
    assert (events["delta"] > 0).any() and events["trigger_text"].notna().any()


def test_video_list_meta_and_segments(conn):
    videos = data.video_list(conn)
    assert len(videos) == 2
    meta = data.session_meta(conn, "video:video_001_vacuum_review")
    assert meta["title"].startswith("무선청소기")
    segments = data.video_segments(meta)
    assert segments["name"].iloc[0] == "0 도입"
    assert data.session_meta(conn, "video:none") == {}


def _render(chart):
    spec = chart.to_dict()
    assert spec  # 직렬화 가능해야 Streamlit이 그린다
    return spec


def test_charts_build(conn):
    from voclens.cx import report as rr
    from voclens.text.taxonomy import Taxonomy

    rdata = rr.load_report_data(conn)
    pains = pd.DataFrame(rr.pain_points(rdata, Taxonomy.load()))
    _render(charts.priority_quadrant(pains))
    journey = pd.DataFrame(rr.journey_table(rdata, 30))
    spec = _render(charts.diverging_bars(journey, "stage", "valence", "t", "n"))
    assert spec["height"] == {"step": 30}            # 범주당 높이 (라벨이 겹쳐 숨지 않게)

    units = data.session_units(conn, "call:call_001_missing_item")
    _render(charts.emotion_timeline(units, data.session_events(conn, "call:call_001_missing_item")))

    vunits = data.session_units(conn, "video:video_002_chicken_mukbang")
    vspec = _render(charts.video_timeline(vunits, data.video_segments(data.session_meta(conn, "video:video_002_chicken_mukbang"))))
    assert vspec["resolve"]["scale"]["color"] == "independent"

    estimates, _, mentions = run_cx(conn, load_settings())
    hspec = _render(charts.share_heatmap(compare.complaint_mix(mentions), compare.CHANNEL_LABELS))
    assert hspec["encoding"]["x"]["sort"][0] == "리뷰"


def test_quadrant_spreads_points_with_same_difficulty():
    pains = pd.DataFrame({
        "label": ["a", "b", "c"], "difficulty": [2, 2, 2], "priority": [90, 50, 10],
        "share": [0.1, 0.1, 0.1], "quadrant": ["x"] * 3,
    })
    datasets = charts.priority_quadrant(pains).to_dict()["datasets"].values()
    points = next(rows for rows in datasets if rows and "label" in rows[0])   # 중앙값 선 데이터는 건너뛴다
    assert len({round(v["x"], 3) for v in points}) == 3


@pytest.mark.parametrize("page", ["홈", "직접 분석", "분석 기준"])
def test_app_pages_run_without_errors(sample_db, page, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("VOCLENS_DASHBOARD_ADMIN", "1")   # 테스트 DB로 바꾸려면 관리 모드가 필요하다
    at = testing.AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.text_input(key="db_path").set_value(str(sample_db)).run()
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("추정치" in w.value for w in at.warning)          # 고지 문구가 모든 화면에 있다


def test_app_empty_db_shows_guide(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("VOCLENS_DASHBOARD_ADMIN", "1")
    at = testing.AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.text_input(key="db_path").set_value(str(tmp_path / "empty.db")).run()
    assert not at.exception
    assert any("분석된 데이터가 없습니다" in i.value for i in at.info)


def test_public_mode_hides_db_controls(monkeypatch):
    """공개 배포(관리 모드 아님)에서는 DB 경로 입력과 다시 분석 버튼이 없어야 한다."""
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.delenv("VOCLENS_DASHBOARD_ADMIN", raising=False)
    at = testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.sidebar.text_input) == 0
    assert not any(b.label == "합성 샘플로 다시 분석" for b in at.sidebar.button)


def test_home_shows_example_result_tabs(sample_db, monkeypatch):
    """리뷰·상담·영상·채널 비교는 홈 안의 탭으로 보여준다 (직접 분석 결과가 어떻게 나오는지 보여주는 예시)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("VOCLENS_DASHBOARD_ADMIN", "1")
    at = testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.text_input(key="db_path").set_value(str(sample_db)).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:5] == ["요약", "리뷰", "상담", "영상", "채널 비교"]
    assert at.sidebar.radio(key="page").options == ["홈", "직접 분석", "분석 기준"]
