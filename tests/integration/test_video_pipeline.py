"""영상 파이프라인 통합 테스트: 번들 JSON → 구간·댓글·지수 → DB → SQL."""

import json

import pytest

from voclens.cli import main
from voclens.config import REPO_ROOT, load_settings
from voclens.ingest.youtube import load_bundle
from voclens.pipelines.video import VideoModels, build_video_session, run_video_pipeline
from voclens.store import db

VIDEOS = REPO_ROOT / "data" / "samples" / "videos"


@pytest.fixture(scope="module")
def sessions():
    settings, models = load_settings(), VideoModels.default()
    return {p.stem: build_video_session(load_bundle(p), models, settings) for p in sorted(VIDEOS.glob("*.json"))}


def test_video_type_and_segments(sessions):
    vacuum = sessions["video_001_vacuum_review"]
    assert vacuum.meta["video_type"] == "review"
    labels = [s["label"] for s in vacuum.meta["segments"]]
    assert labels[0] == "도입" and labels[-1] == "총평"
    assert {"언박싱", "사용", "장단점"} <= set(labels)


def test_boilerplate_removed(sessions):
    texts = [u.text for u in sessions["video_001_vacuum_review"].units]
    assert not any("구독" in t or "끝까지 봐" in t for t in texts)


def test_product_video_uses_product_taxonomy(sessions):
    labels = {l for u in sessions["video_001_vacuum_review"].units for l in u.complaint_categories}
    assert {"배터리", "성능", "가격"} <= labels
    assert "배달 지연" not in labels


def test_main_speaker_and_guest(sessions):
    roles = {u.speaker_role for u in sessions["video_002_chicken_mukbang"].units}
    assert {"creator", "guest", "commenter"} <= roles


def test_timestamp_comments_linked_to_segments(sessions):
    vacuum = sessions["video_001_vacuum_review"]
    stamped = [u for u in vacuum.units if u.speaker_role == "commenter" and u.start_sec is not None]
    assert len(stamped) == vacuum.meta["n_timestamp_comments"] == 6
    battery = next(u for u in stamped if "배터리" in u.text)
    assert battery.segment == "사용" and battery.valence < 0
    assert all(u.segment_index is not None for u in stamped)


def test_indices_and_audience(sessions):
    for s in sessions.values():
        assert s.indices["engagement"].raw < 100      # 시청자 반응 기준 (리뷰 공식처럼 포화되지 않음)
        assert s.meta["engagement_basis"] == "audience"
        assert set(s.meta["audience_indices"]) == {"stress", "happiness", "negative_affect", "frustration", "engagement"}
        assert -1 <= s.meta["weighted_valence"] <= 1


def test_bundle_without_comments_or_views(tmp_path):
    data = json.loads((VIDEOS / "video_002_chicken_mukbang.json").read_text(encoding="utf-8"))
    data["comments"] = []
    data["meta"].pop("views")
    (tmp_path / "v.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    s = build_video_session(load_bundle(tmp_path / "v.json"), VideoModels.default(), load_settings())
    assert s.meta["n_comments"] == 0 and "audience_indices" not in s.meta
    assert s.meta.get("engagement_basis") is None


def test_run_pipeline_and_queries(tmp_path):
    path = tmp_path / "v.db"
    settings = load_settings()
    run_video_pipeline([str(VIDEOS)], settings, db_path=path)
    run_video_pipeline([str(VIDEOS)], settings, db_path=path)
    conn = db.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE source_type = 'video'").fetchone()[0] == 2
    src = conn.execute("SELECT title, url, brand, created_at FROM sources WHERE source_id = 'video_001_vacuum_review'").fetchone()
    assert src["title"].startswith("무선청소기") and src["created_at"] == "2026-09-20"
    for name in ("video_overview", "video_segments", "video_comment_complaints"):
        assert db.run_query(conn, name)


def test_migration_adds_segment_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE units (unit_id TEXT PRIMARY KEY, session_id TEXT, source_type TEXT, ord INTEGER, text TEXT)")
    old.commit()
    old.close()
    conn = db.connect(path)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(units)")}
    assert {"segment", "segment_index"} <= columns


def test_cli_video(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "cli.db"), "video", str(VIDEOS)]) == 0
    out = capsys.readouterr().out
    assert "video_001_vacuum_review" in out and "총평" in out
