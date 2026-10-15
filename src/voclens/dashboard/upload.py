"""대시보드 업로드 분석: 방문자가 올린 리뷰·상담·영상을 그 자리에서 분석한다.

- 업로드한 데이터는 방문자마다 따로 만든 메모리 DB에만 넣는다. 공유 샘플 DB와 섞이지 않고 디스크에 저장하지 않는다.
- 공개 배포에서 서버가 버틸 수 있도록 크기를 제한한다 (LIMITS).
- 음성 파일만 있고 전사본이 없으면 STT가 필요하다. STT 모델(faster-whisper)은 배포본에 넣지 않았으므로
  설치된 환경(로컬 `pip install -e .[audio]`)에서만 동작한다.
"""

from __future__ import annotations

import copy
import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from voclens.cx.cx_report import run_cx
from voclens.ingest.call_transcript import CallTranscript, load_transcript, parse_text_transcript
from voclens.ingest.review_csv import ReviewRecord
from voclens.ingest.subtitles import cues_to_sentences, parse_subtitles
from voclens.ingest.youtube import VideoBundle, VideoComment, VideoSegment, fetch_bundle
from voclens.pipelines.call import CallModels, build_call_session
from voclens.pipelines.review import ReviewModels, analyze_records
from voclens.pipelines.video import VideoModels, build_video_session
from voclens.schema import Session
from voclens.store import db

LIMITS = {
    "review_rows": 5000,
    "calls": 20,
    "comments": 500,
}


@dataclass
class UploadOutcome:
    sessions: list[Session]
    notes: list[str] = field(default_factory=list)


def new_db():
    """방문자 전용 메모리 DB."""
    return db.connect(":memory:", check_same_thread=False)


def read_table(data: bytes, filename: str = "") -> pd.DataFrame:
    """CSV/TSV를 UTF-8 또는 CP949로 읽는다 (엑셀에서 저장한 한글 CSV 대응)."""
    sep = "\t" if filename.lower().endswith((".tsv", ".txt")) else ","
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일을 UTF-8 또는 CP949로 읽을 수 없습니다. CSV로 다시 저장해 주세요.")


def guess_text_column(df: pd.DataFrame) -> str:
    """리뷰 본문 열 추정: 흔한 이름이 있으면 그 열, 없으면 평균 글자 수가 가장 긴 열."""
    for name in ("text", "review", "content", "리뷰", "내용", "본문", "comment", "댓글"):
        for col in df.columns:
            if col.strip().lower() == name:
                return col
    return max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())


def _save(conn, sessions: list[Session], pipeline: str, models_versions: dict, taxonomy) -> None:
    run_id = db.start_run(conn, pipeline, "upload", {}, models_versions)
    db.save_sessions(
        conn, sessions, run_id, label_source="keyword",
        category_of=lambda label: taxonomy.category_of(label) if taxonomy.is_complaint(label) else None,
    )
    db.finish_run(conn, run_id, len(sessions))


# --- 리뷰 ---

def analyze_reviews(conn, df: pd.DataFrame, columns: dict[str, str | None], domain: str, settings: dict) -> UploadOutcome:
    notes = []
    if len(df) > LIMITS["review_rows"]:
        notes.append(f"리뷰가 {len(df):,}건이라 앞의 {LIMITS['review_rows']:,}건만 분석했습니다.")
        df = df.head(LIMITS["review_rows"])

    def get(row, key):
        col = columns.get(key)
        value = str(row[col]).strip() if col and col in df.columns else ""
        return value or None

    records = []
    for i, row in df.iterrows():
        text = get(row, "text")
        if not text:
            continue
        rating = get(row, "rating")
        try:
            rating_value = float(rating) if rating else None
        except ValueError:
            rating_value = None
        records.append(
            ReviewRecord(
                source_id=f"upload:{i}", text=text, rating=rating_value, created_at=get(row, "created_at"),
                brand=get(row, "brand") or "업로드", store=get(row, "store"),
            )
        )
    if not records:
        raise ValueError("본문이 있는 리뷰가 없습니다. 리뷰 본문 열을 확인해 주세요.")

    local = copy.deepcopy(settings)
    local["review"]["domain"] = domain
    models = ReviewModels.default()
    sessions = analyze_records(records, local, models)
    _save(conn, sessions, "upload-review", models.versions(), models.taxonomy.taxonomy)
    rated = sum(r.rating is not None for r in records)
    if rated < 20:
        notes.append(f"별점이 있는 리뷰가 {rated}건이라(20건 미만) CSAT 추정 모델은 학습하지 않았습니다. 감정·불만 분석은 그대로 합니다.")
    return UploadOutcome(sessions, notes)


# --- 상담 ---

def load_call_upload(name: str, data: bytes) -> CallTranscript:
    """전사본 업로드 하나를 읽는다: .json(전사본 형식) 또는 .txt('화자: 발화' 줄)."""
    stem = Path(name).stem or "call"
    if name.lower().endswith(".json"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{stem}.json"
            path.write_bytes(data)
            call = load_transcript(path)
        call.audio = None
        call.source_path = None
        return call
    return parse_text_transcript(data.decode("utf-8-sig", errors="replace"), call_id=stem)


def analyze_calls(conn, calls: list[CallTranscript], audio: dict[str, bytes], settings: dict, domain: str | None = None) -> UploadOutcome:
    """audio: 전사본 call_id(파일 이름 앞부분)와 같은 이름의 WAV 바이트. 있으면 음향 분석을 더한다."""
    notes = []
    if len(calls) > LIMITS["calls"]:
        notes.append(f"전사본이 {len(calls)}개라 앞의 {LIMITS['calls']}개만 분석했습니다.")
        calls = calls[: LIMITS["calls"]]
    models = CallModels.default(settings)
    sessions = []
    with tempfile.TemporaryDirectory() as tmp:
        for call in calls:
            if domain:
                call.meta.setdefault("domain", domain)
            wav = audio.get(call.call_id)
            if wav is not None:
                path = Path(tmp) / f"{call.call_id}.wav"
                path.write_bytes(wav)
                call.audio = str(path)
            sessions.append(build_call_session(call, models, settings))
    unmatched = set(audio) - {c.call_id for c in calls}
    if unmatched:
        notes.append(f"이름이 맞는 전사본이 없어 쓰지 않은 음성 파일: {', '.join(sorted(unmatched))}")
    _save(conn, sessions, "upload-call", models.versions(), models.taxonomy.taxonomy)
    return UploadOutcome(sessions, notes)


def transcribe_audio_only(name: str, data: bytes) -> CallTranscript:
    """전사본 없이 음성만 올렸을 때. STT 의존성이 없으면 RuntimeError로 안내한다."""
    from voclens.audio.stt import audio_to_transcript

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        return CallTranscript.model_validate(audio_to_transcript(path))


# --- 영상 ---

def read_comments(data: bytes, filename: str) -> list[VideoComment]:
    """댓글 파일: CSV(text/댓글 열, 선택 likes 열) 또는 한 줄에 댓글 하나인 TXT."""
    if filename.lower().endswith(".csv"):
        df = read_table(data, filename)
        col = guess_text_column(df)
        likes = next((c for c in df.columns if c.strip().lower() in ("likes", "like", "좋아요")), None)
        return [
            VideoComment(text=str(r[col]).strip(), likes=int(float(r[likes])) if likes and str(r[likes]).strip() else 0)
            for _, r in df.iterrows()
            if str(r[col]).strip()
        ][: LIMITS["comments"]]
    lines = data.decode("utf-8-sig", errors="replace").splitlines()
    return [VideoComment(text=l.strip()) for l in lines if l.strip()][: LIMITS["comments"]]


def bundle_from_subtitles(subtitle: bytes, comments: list[VideoComment], meta: dict, video_id: str = "uploaded_video") -> VideoBundle:
    cues = parse_subtitles(subtitle.decode("utf-8-sig", errors="replace"))
    sentences = cues_to_sentences(cues)
    if not sentences:
        raise ValueError("자막에서 문장을 찾지 못했습니다. VTT 또는 SRT 형식인지 확인해 주세요.")
    return VideoBundle(
        video_id=video_id,
        meta=meta,
        segments=[VideoSegment(start=s.start, end=s.end, text=s.text) for s in sentences],
        comments=comments,
        transcript_source="uploaded_subtitles",
    )


def bundle_from_json(data: bytes) -> VideoBundle:
    raw = json.loads(data.decode("utf-8-sig"))
    raw.pop("audio", None)          # 업로드 JSON이 서버의 다른 파일 경로를 가리키지 못하게
    raw.pop("source_path", None)
    raw.setdefault("video_id", "uploaded_video")
    return VideoBundle.model_validate(raw)


def bundle_from_url(url: str) -> VideoBundle:
    """YouTube URL. yt-dlp가 설치된 환경에서만 동작한다."""
    return fetch_bundle(url, max_comments=LIMITS["comments"])


def analyze_video(conn, bundle: VideoBundle, settings: dict) -> UploadOutcome:
    models = VideoModels.default()
    notes = []
    if not bundle.comments:
        notes.append("댓글이 없어 시청자 감정·타임스탬프 댓글 분석은 하지 않았습니다.")
    if not bundle.meta.get("views"):
        notes.append("조회수가 없어 몰입 지수는 시청자 반응 대신 자막 내용으로 계산했습니다.")
    session = build_video_session(bundle, models, settings)
    _save(conn, [session], "upload-video", models.versions(), models.taxonomy.taxonomy)
    return UploadOutcome([session], notes)


def estimate(conn, settings: dict):
    """업로드 DB의 CSAT·CES·NPS 추정 (별점 리뷰가 20건 미만이면 CSAT는 비워 둔다)."""
    return run_cx(conn, settings)
