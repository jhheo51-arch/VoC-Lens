"""리뷰 파이프라인 통합 테스트: CSV → DB → SQL 쿼리 → 리포트."""

import csv

import pytest

from voclens.cli import main
from voclens.config import REPO_ROOT, load_settings
from voclens.cx.report import build_report
from voclens.pipelines.call import run_call_pipeline
from voclens.pipelines.video import run_video_pipeline
from voclens.pipelines.review import build_session, run_review_pipeline
from voclens.ingest.review_csv import ReviewRecord, load_reviews
from voclens.store import db

SAMPLE = REPO_ROOT / "data" / "samples" / "delivery_reviews_synthetic.csv"


@pytest.fixture(scope="module")
def pipeline_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    settings = load_settings()
    run_id, sessions = run_review_pipeline(SAMPLE, settings, db_path=path)
    # 상담·영상 쿼리도 함께 검사하려고 같은 DB에 샘플을 넣는다
    run_call_pipeline([REPO_ROOT / "data" / "samples" / "calls"], settings, db_path=path)
    run_video_pipeline([str(REPO_ROOT / "data" / "samples" / "videos")], settings, db_path=path)
    return path, run_id, sessions, settings


def test_pipeline_stores_every_review(pipeline_db):
    path, run_id, sessions, _ = pipeline_db
    conn = db.connect(path)
    n_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE source_type = 'review'").fetchone()[0]
    run = conn.execute("SELECT status, n_sessions FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    with SAMPLE.open(encoding="utf-8-sig") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    assert n_sessions == len(sessions) == n_rows
    assert tuple(run) == ("done", n_rows)


def test_pipeline_rerun_replaces_instead_of_duplicating(pipeline_db):
    path, _, sessions, settings = pipeline_db
    run_review_pipeline(SAMPLE, settings, db_path=path)
    conn = db.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE source_type = 'review'").fetchone()[0] == len(sessions)
    assert conn.execute("SELECT COUNT(*) FROM units WHERE source_type = 'review'").fetchone()[0] == sum(
        len(s.units) for s in sessions
    )


@pytest.mark.parametrize("name", db.list_queries())
def test_every_saved_query_runs(pipeline_db, name):
    conn = db.connect(pipeline_db[0])
    rows = db.run_query(conn, name)
    assert len(rows) > 0


def test_complaint_share_finds_designed_brand_issues(pipeline_db):
    """합성 데이터는 A=지연·식음, B=누락이 주요 불만이 되도록 만들었다."""
    conn = db.connect(pipeline_db[0])
    rows = [dict(r) for r in db.run_query(conn, "complaint_share_by_brand")]
    top = {}
    for r in rows:
        top.setdefault(r["brand"], []).append(r["label"])
    assert {"배달 지연", "온도·식감"} & set(top["치킨브랜드A"][:3])
    assert top["피자브랜드B"][0] == "누락·오배송"


def test_report_renders_all_sections(pipeline_db):
    path, _, _, settings = pipeline_db
    conn = db.connect(path)
    text = build_report(conn, settings)
    for heading in ["## 1. 한눈에 보기", "## 2. 우선 개선 과제", "## 3. 브랜드 비교", "## 6. 별점–감정 불일치 리뷰"]:
        assert heading in text
    assert "Quick Win" in text or "핵심 과제" in text


def test_report_filters(pipeline_db):
    path, _, _, settings = pipeline_db
    conn = db.connect(path)
    text = build_report(conn, settings, brand="피자브랜드B", exclude_review_event=True)
    assert "브랜드 = 피자브랜드B" in text
    assert "리뷰 이벤트 = 제외" in text


def test_review_event_clause_excluded_but_flagged():
    session = build_session(ReviewRecord(source_id="t:0", text="리뷰이벤트 참여합니다! 치킨이 식었어요"))
    assert "review_event" in session.flags
    assert [u.text for u in session.units] == ["치킨이 식었어요"]


def test_pii_masked_in_stored_text():
    session = build_session(ReviewRecord(source_id="t:1", text="010-1234-5678로 연락 주세요 배달이 늦었어요"))
    assert "010" not in session.text
    assert all("010" not in u.text for u in session.units)


def test_load_reviews_requires_text_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("content\n맛있어요\n", encoding="utf-8")
    with pytest.raises(ValueError, match="리뷰 본문 열"):
        load_reviews(path, {"text": "text"})


def test_load_reviews_reads_cp949(tmp_path):
    path = tmp_path / "cp949.csv"
    path.write_bytes("text,rating\n맛있어요,5\n,3\n".encode("cp949"))
    records = load_reviews(path, {"text": "text", "rating": "rating"})
    assert len(records) == 1 and records[0].rating == 5.0


def test_cli_end_to_end(tmp_path, capsys):
    path = tmp_path / "cli.db"
    out = tmp_path / "report.md"
    assert main(["--db", str(path), "review", str(SAMPLE)]) == 0
    assert main(["--db", str(path), "report", "--out", str(out)]) == 0
    assert main(["--db", str(path), "query", "index_by_brand"]) == 0
    assert out.read_text(encoding="utf-8").startswith("# 리뷰 VoC 개선 과제 리포트")
    assert "치킨브랜드A" in capsys.readouterr().out
