-- VoC Lens SQLite 스키마
-- 리뷰·상담·영상이 같은 테이블을 쓰고 source_type으로 구분한다.

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    pipeline        TEXT NOT NULL,
    input_path      TEXT,
    config_json     TEXT,
    models_json     TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,
    n_sessions      INTEGER,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,          -- review | call | video
    brand           TEXT,
    store           TEXT,
    title           TEXT,
    url             TEXT,
    created_at      TEXT,
    rating          REAL,
    meta_json       TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES sources(source_id),
    source_type     TEXT NOT NULL,
    run_id          TEXT REFERENCES runs(run_id),
    text            TEXT,                   -- 개인정보를 가린 원문
    flags_json      TEXT,                   -- ["review_event", ...]
    peak_end        REAL
);

CREATE TABLE IF NOT EXISTS units (
    unit_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    source_type     TEXT NOT NULL,
    ord             INTEGER NOT NULL,
    speaker_role    TEXT,
    start_sec       REAL,
    end_sec         REAL,
    text            TEXT NOT NULL,
    valence         REAL,
    arousal         REAL,
    emotion_json    TEXT,
    aspect          TEXT,
    aspect_sentiment REAL,
    journey_stage   TEXT,
    segment         TEXT,
    segment_index   INTEGER,
    cx_signals_json TEXT,
    scores_json     TEXT,
    confidence_json TEXT
);

-- 사람·모델·규칙 라벨을 같은 표에 두고 source로 구분한다 (학습 데이터 추출, 성능 비교용)
CREATE TABLE IF NOT EXISTS unit_labels (
    unit_id         TEXT NOT NULL REFERENCES units(unit_id),
    label_type      TEXT NOT NULL,          -- complaint | journey | emotion ...
    label           TEXT NOT NULL,
    category        TEXT,                   -- 불만 유형의 대분류
    source          TEXT NOT NULL,          -- keyword | zero_shot | model | human
    PRIMARY KEY (unit_id, label_type, label, source)
);

CREATE TABLE IF NOT EXISTS indices (
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    index_name      TEXT NOT NULL,
    raw             REAL,
    mean            REAL,
    peak            REAL,
    above_ratio     REAL,
    end_value       REAL,
    percentile      REAL,
    PRIMARY KEY (session_id, index_name)
);

-- 감정 급변 구간 (상담·영상, 4주차 이후)
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    start_sec       REAL,
    end_sec         REAL,
    index_name      TEXT,
    delta           REAL,
    trigger_unit_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_units_session ON units(session_id);
CREATE INDEX IF NOT EXISTS idx_labels_type ON unit_labels(label_type, label);
CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source_id);
