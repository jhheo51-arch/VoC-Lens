"""대시보드용 데이터 조회 (Streamlit 없이 테스트할 수 있게 분리).

모든 함수는 sqlite 연결을 받아 pandas DataFrame이나 dict를 돌려준다.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from voclens.config import REPO_ROOT

SAMPLES = REPO_ROOT / "data" / "samples"
ROLE_LABELS = {"customer": "고객", "agent": "상담원", "creator": "크리에이터", "guest": "게스트", "commenter": "댓글"}
OUTCOME_LABELS = {"resolved": "해결", "pending": "보류", "transferred": "이관", "unknown": "불명"}


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT source_type, COUNT(*) FROM sessions GROUP BY source_type").fetchall()
    return {r[0]: r[1] for r in rows}


def has_estimates(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0] > 0


def build_sample_db(db_path: str | Path, settings: dict) -> None:
    """합성 샘플 데이터로 세 채널 분석과 CX 추정을 한 번에 실행한다 (데모·배포용)."""
    from voclens.cx.cx_report import run_cx
    from voclens.pipelines.call import run_call_pipeline
    from voclens.pipelines.review import run_review_pipeline
    from voclens.pipelines.video import run_video_pipeline
    from voclens.store import db

    run_review_pipeline(SAMPLES / "delivery_reviews_synthetic.csv", settings, db_path=db_path)
    run_call_pipeline([SAMPLES / "calls"], settings, db_path=db_path)
    run_video_pipeline([str(SAMPLES / "videos")], settings, db_path=db_path)
    conn = db.connect(db_path)
    try:
        run_cx(conn, settings)
    finally:
        conn.close()


def brands(conn: sqlite3.Connection, source_type: str = "review") -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT src.brand FROM sessions s JOIN sources src ON src.source_id = s.source_id"
        " WHERE s.source_type = ? AND src.brand IS NOT NULL ORDER BY src.brand",
        (source_type,),
    ).fetchall()
    return [r[0] for r in rows]


# --- 상담 ---

def call_list(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT s.session_id, s.source_id AS call_id, src.meta_json, s.peak_end,
               MAX(CASE WHEN i.index_name = 'frustration' THEN i.raw END) AS frustration,
               MAX(CASE WHEN i.index_name = 'stress' THEN i.raw END) AS stress,
               MAX(CASE WHEN i.index_name = 'recovery' THEN i.raw END) AS recovery,
               e.ces_est, e.csat_est
        FROM sessions s
        JOIN sources src ON src.source_id = s.source_id
        JOIN indices i ON i.session_id = s.session_id
        LEFT JOIN estimates e ON e.session_id = s.session_id
        WHERE s.source_type = 'call'
        GROUP BY s.session_id
        ORDER BY frustration DESC
        """,
        conn,
    )
    meta = df.pop("meta_json").map(json.loads)
    df["outcome"] = meta.map(lambda m: OUTCOME_LABELS.get(m.get("outcome"), m.get("outcome")))
    for key in ("duration", "customer_interruptions", "repeat_count", "max_silence", "scenario"):
        df[key] = meta.map(lambda m, k=key: m.get(k))
    return df


def session_units(conn: sqlite3.Connection, session_id: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT unit_id, ord, speaker_role, start_sec, end_sec, text, valence, arousal, segment, segment_index,"
        " aspect, scores_json, cx_signals_json FROM units WHERE session_id = ? ORDER BY ord",
        conn,
        params=(session_id,),
    )
    scores = df.pop("scores_json").map(json.loads).apply(pd.Series)
    df = pd.concat([df, scores], axis=1)
    df["signals"] = df.pop("cx_signals_json").map(lambda s: ", ".join(sorted(json.loads(s))))
    df["role"] = df["speaker_role"].map(ROLE_LABELS).fillna(df["speaker_role"])
    labels = pd.read_sql_query(
        "SELECT l.unit_id, group_concat(l.label, ', ') AS complaints FROM unit_labels l"
        " JOIN units u ON u.unit_id = l.unit_id WHERE u.session_id = ? AND l.label_type = 'complaint'"
        " GROUP BY l.unit_id",
        conn,
        params=(session_id,),
    )
    return df.merge(labels, on="unit_id", how="left")


def session_events(conn: sqlite3.Connection, session_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT e.index_name, e.start_sec, e.end_sec, e.delta, t.text AS trigger_text"
        " FROM events e LEFT JOIN units t ON t.unit_id = e.trigger_unit_id WHERE e.session_id = ? ORDER BY e.start_sec",
        conn,
        params=(session_id,),
    )


def session_meta(conn: sqlite3.Connection, session_id: str) -> dict:
    row = conn.execute(
        "SELECT src.meta_json, src.title, src.brand, src.url, src.created_at FROM sessions s"
        " JOIN sources src ON src.source_id = s.source_id WHERE s.session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        return {}
    meta = json.loads(row["meta_json"] or "{}")
    meta.update({k: row[k] for k in ("title", "brand", "url", "created_at") if row[k] is not None})
    return meta


def session_indices(conn: sqlite3.Connection, session_id: str) -> dict[str, float | None]:
    rows = conn.execute("SELECT index_name, raw FROM indices WHERE session_id = ?", (session_id,)).fetchall()
    return {r[0]: r[1] for r in rows}


# --- 영상 ---

def video_list(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT s.session_id, s.source_id AS video_id, src.title, src.brand, src.meta_json FROM sessions s"
        " JOIN sources src ON src.source_id = s.source_id WHERE s.source_type = 'video' ORDER BY src.created_at DESC",
        conn,
    )
    meta = df.pop("meta_json").map(json.loads)
    df["video_type"] = meta.map(lambda m: m.get("video_type_label"))
    df["weighted_valence"] = meta.map(lambda m: m.get("weighted_valence"))
    df["n_comments"] = meta.map(lambda m: m.get("n_comments"))
    return df


def video_segments(meta: dict) -> pd.DataFrame:
    df = pd.DataFrame(meta.get("segments", []))
    if not df.empty:
        df["name"] = df["index"].astype(str) + " " + df["label"]
    return df
