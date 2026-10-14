"""리뷰 파이프라인: CSV → 정제 → 절 분리 → 감정 → 불만 유형·속성 → CX 신호 → 여정 단계 → 지수 → DB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicecx.indices.compute import assign_percentiles, score_session
from voicecx.ingest.review_csv import ReviewRecord, load_reviews
from voicecx.schema import Session, SourceType, Unit
from voicecx.store import db
from voicecx.text.cx_signals import CxSignalDetector
from voicecx.text.emotion import LexiconEmotionModel
from voicecx.text.journey import JourneyTagger
from voicecx.text.preprocess import detect_review_event, mask_pii, preprocess
from voicecx.text.segment import segment
from voicecx.text.taxonomy import KeywordTaxonomyClassifier


@dataclass
class ReviewModels:
    emotion: LexiconEmotionModel
    taxonomy: KeywordTaxonomyClassifier
    cx: CxSignalDetector
    journey: JourneyTagger

    @classmethod
    def default(cls) -> ReviewModels:
        taxonomy = KeywordTaxonomyClassifier()
        return cls(
            emotion=LexiconEmotionModel(),
            taxonomy=taxonomy,
            cx=CxSignalDetector(),
            journey=JourneyTagger(taxonomy=taxonomy.taxonomy),
        )

    def versions(self) -> dict[str, str]:
        return {k: getattr(self, k).name for k in ("emotion", "taxonomy", "cx", "journey")}


def build_session(record: ReviewRecord) -> Session:
    pre = preprocess(record.text)
    session_id = f"review:{record.source_id}"
    # "리뷰이벤트 참여합니다" 같은 절은 경험에 대한 평가가 아니라서 지수 계산에서 뺀다 (리뷰에는 표시가 남는다)
    clauses = [c for c in segment(pre.text) if not detect_review_event(c)]
    units = [
        Unit(
            unit_id=f"{session_id}#{i}",
            session_id=session_id,
            source_type=SourceType.REVIEW,
            order=i,
            text=clause,
            clean_text=clause,
            speaker_role="customer",
        )
        for i, clause in enumerate(clauses)
    ]
    meta = {
        "rating": record.rating,
        "created_at": record.created_at,
        "brand": record.brand,
        "store": record.store,
    }
    if record.owner_reply:
        meta["owner_reply"] = mask_pii(record.owner_reply)
    return Session(
        session_id=session_id,
        source_id=record.source_id,
        source_type=SourceType.REVIEW,
        text=mask_pii(record.text),
        meta=meta,
        flags=pre.flags,
        units=units,
    )


def analyze_records(records: list[ReviewRecord], settings: dict, models: ReviewModels | None = None) -> list[Session]:
    models = models or ReviewModels.default()
    idx = settings["indices"]
    sessions = [build_session(r) for r in records]
    for session in sessions:
        units = session.units
        # 순서가 중요하다: 불만 분류는 Valence를 쓰고, 명시적 불만이면 Valence를 낮춘다
        models.emotion.predict(units)
        models.taxonomy.predict(units, domain=settings["review"].get("domain"))
        models.cx.predict(units)
        models.journey.predict(units)
        score_session(session, threshold=idx["above_threshold"], min_negative=idx["recovery_min_negative"])
    assign_percentiles(sessions)
    return sessions


def run_review_pipeline(csv_path: str | Path, settings: dict, db_path: str | Path | None = None) -> tuple[str, list[Session]]:
    models = ReviewModels.default()
    records = load_reviews(csv_path, settings["review"]["columns"])
    conn = db.connect(db_path or settings["db_path"])
    run_id = db.start_run(conn, "review", str(csv_path), settings, models.versions())
    try:
        sessions = analyze_records(records, settings, models)
        tx = models.taxonomy.taxonomy
        db.save_sessions(
            conn, sessions, run_id, label_source="keyword",
            # 칭찬·분류 불가는 불만이 아니므로 대분류를 비워 둔다 (SQL에서 category IS NOT NULL로 불만만 고름)
            category_of=lambda label: tx.category_of(label) if tx.is_complaint(label) else None,
        )
        db.finish_run(conn, run_id, len(sessions))
    except Exception as exc:
        db.finish_run(conn, run_id, 0, error=repr(exc))
        raise
    finally:
        conn.close()
    return run_id, sessions
