"""채널 통합 CX 리포트 통합 테스트: 리뷰·상담·영상 → CSAT·CES·NPS 추정 → 채널 비교."""

import pytest

from voclens.cli import main
from voclens.config import REPO_ROOT, load_settings
from voclens.cx.cx_report import build_cx_report, run_cx
from voclens.pipelines.call import run_call_pipeline
from voclens.pipelines.review import run_review_pipeline
from voclens.pipelines.video import run_video_pipeline
from voclens.store import db

SAMPLES = REPO_ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def all_channels_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("cx") / "cx.db"
    settings = load_settings()
    run_review_pipeline(SAMPLES / "delivery_reviews_synthetic.csv", settings, db_path=path)
    run_call_pipeline([SAMPLES / "calls"], settings, db_path=path)
    run_video_pipeline([str(SAMPLES / "videos")], settings, db_path=path)
    return path, settings


def test_estimates_saved_for_every_session(all_channels_db):
    path, settings = all_channels_db
    conn = db.connect(path)
    run_cx(conn, settings)
    n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    rows = conn.execute("SELECT csat_est, ces_est, nps_group FROM estimates").fetchall()
    assert len(rows) == n_sessions
    assert all(1 <= r["csat_est"] <= 5 and 1 <= r["ces_est"] <= 7 for r in rows)
    assert {r["nps_group"] for r in rows} <= {"promoter", "detractor", "unknown"}
    assert db.run_query(conn, "estimates_by_channel")


def test_csat_model_beats_baselines(all_channels_db):
    path, settings = all_channels_db
    _, evaluation, _ = run_cx(db.connect(path), settings)
    assert evaluation.mae_model < evaluation.mae_valence_only < evaluation.mae_constant
    assert abs(evaluation.csat_pct_est - evaluation.csat_pct_actual) <= abs(evaluation.csat_pct_est_raw - evaluation.csat_pct_actual)


def test_frustrating_call_has_highest_ces(all_channels_db):
    path, settings = all_channels_db
    estimates, _, _ = run_cx(db.connect(path), settings)
    calls = estimates[estimates["source_type"] == "call"].set_index("session_id")
    assert calls["ces_est"].idxmax() == "call:call_002_late_delivery"


def test_domain_filter(all_channels_db):
    path, settings = all_channels_db
    conn = db.connect(path)
    delivery, _, mentions = run_cx(conn, settings, domain="delivery")
    assert "video:video_001_vacuum_review" not in set(delivery["session_id"])
    assert "배터리" not in set(mentions["label"])
    everything, _, _ = run_cx(conn, settings, domain=None)
    assert "video:video_001_vacuum_review" in set(everything["session_id"])


def test_report_sections(all_channels_db):
    path, settings = all_channels_db
    text = build_cx_report(db.connect(path), settings)
    for heading in ["## 1. 채널별 CX 지표", "## 2. CSAT 추정 모델 검증", "## 3. 불만 유형 구성비", "조용한 불만"]:
        assert heading in text
    assert "도메인: 배달" in text
    assert "| 리뷰 |" in text and "| 상담 |" in text


def test_rerun_pipeline_after_estimates(all_channels_db):
    """estimates가 있는 세션을 다시 저장해도 외래 키 오류가 나지 않아야 한다."""
    path, settings = all_channels_db
    run_cx(db.connect(path), settings)
    run_call_pipeline([SAMPLES / "calls"], settings, db_path=path)


def test_cli_cx(all_channels_db, tmp_path):
    path, _ = all_channels_db
    out = tmp_path / "cx.md"
    assert main(["--db", str(path), "cx", "--out", str(out), "--domain", "all"]) == 0
    assert "도메인 구분 없음" in out.read_text(encoding="utf-8")
