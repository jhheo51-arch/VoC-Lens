"""SQLite 저장소."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from voicecx.schema import Session

QUERIES_PACKAGE = "voicecx.store"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(resources.files(QUERIES_PACKAGE).joinpath("schema.sql").read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


# 예전 버전으로 만든 DB에 없는 열을 추가한다 (테이블, 열, 타입)
_ADDED_COLUMNS = [("units", "segment", "TEXT"), ("units", "segment_index", "INTEGER")]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, kind in _ADDED_COLUMNS:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()


def start_run(conn: sqlite3.Connection, pipeline: str, input_path: str, config: dict, models: dict) -> str:
    run_id = f"{pipeline}-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    conn.execute(
        "INSERT INTO runs (run_id, pipeline, input_path, config_json, models_json, started_at, status)"
        " VALUES (?, ?, ?, ?, ?, ?, 'running')",
        (run_id, pipeline, input_path, _json(config), _json(models), _now()),
    )
    conn.commit()
    return run_id


def finish_run(conn: sqlite3.Connection, run_id: str, n_sessions: int, error: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ?, status = ?, n_sessions = ?, error = ? WHERE run_id = ?",
        (_now(), "failed" if error else "done", n_sessions, error, run_id),
    )
    conn.commit()


def _delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("DELETE FROM unit_labels WHERE unit_id IN (SELECT unit_id FROM units WHERE session_id = ?)", (session_id,))
    conn.execute("DELETE FROM units WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM indices WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def save_sessions(
    conn: sqlite3.Connection,
    sessions: list[Session],
    run_id: str,
    label_source: str,
    category_of=lambda label: None,
) -> None:
    """세션을 저장한다. 같은 session_id가 있으면 지우고 다시 쓴다 (재실행 가능)."""
    with conn:
        for s in sessions:
            meta = dict(s.meta)
            _delete_session(conn, s.session_id)
            conn.execute(
                "INSERT INTO sources (source_id, source_type, brand, store, title, url, created_at, rating, meta_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(source_id) DO UPDATE SET source_type = excluded.source_type, brand = excluded.brand,"
                " store = excluded.store, title = excluded.title, url = excluded.url, created_at = excluded.created_at,"
                " rating = excluded.rating, meta_json = excluded.meta_json",
                (
                    s.source_id, s.source_type.value, meta.pop("brand", None), meta.pop("store", None),
                    meta.pop("title", None), meta.pop("url", None), meta.pop("created_at", None),
                    meta.pop("rating", None), _json(meta),
                ),
            )
            conn.execute(
                "INSERT INTO sessions (session_id, source_id, source_type, run_id, text, flags_json, peak_end)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (s.session_id, s.source_id, s.source_type.value, run_id, s.text, _json(s.flags), s.peak_end),
            )
            for u in s.units:
                conn.execute(
                    "INSERT INTO units (unit_id, session_id, source_type, ord, speaker_role, start_sec, end_sec, text,"
                    " valence, arousal, emotion_json, aspect, aspect_sentiment, journey_stage, segment, segment_index,"
                    " cx_signals_json, scores_json, confidence_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        u.unit_id, u.session_id, u.source_type.value, u.order, u.speaker_role, u.start_sec, u.end_sec,
                        u.text, u.valence, u.arousal, _json(u.emotion_probs), u.aspect, u.aspect_sentiment,
                        u.journey_stage, u.segment, u.segment_index, _json(u.cx_signals), _json(u.scores),
                        _json(u.confidence),
                    ),
                )
                for label in u.complaint_categories:
                    conn.execute(
                        "INSERT OR IGNORE INTO unit_labels (unit_id, label_type, label, category, source) VALUES (?, 'complaint', ?, ?, ?)",
                        (u.unit_id, label, category_of(label), label_source),
                    )
                if u.journey_stage:
                    conn.execute(
                        "INSERT OR IGNORE INTO unit_labels (unit_id, label_type, label, source) VALUES (?, 'journey', ?, ?)",
                        (u.unit_id, u.journey_stage, label_source),
                    )
            for e in s.events:
                conn.execute(
                    "INSERT INTO events (event_id, session_id, start_sec, end_sec, index_name, delta, trigger_unit_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (e.event_id, s.session_id, e.start_sec, e.end_sec, e.index_name, e.delta, e.trigger_unit_id),
                )
            for idx in s.indices.values():
                conn.execute(
                    "INSERT INTO indices (session_id, index_name, raw, mean, peak, above_ratio, end_value, percentile)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (s.session_id, idx.name, idx.raw, idx.mean, idx.peak, idx.above_ratio, idx.end_value, idx.percentile),
                )


def list_queries() -> list[str]:
    folder = resources.files(QUERIES_PACKAGE).joinpath("queries")
    return sorted(p.name.removesuffix(".sql") for p in folder.iterdir() if p.name.endswith(".sql"))


def read_query(name: str) -> str:
    return resources.files(QUERIES_PACKAGE).joinpath("queries", f"{name}.sql").read_text(encoding="utf-8")


def run_query(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    return conn.execute(read_query(name)).fetchall()
