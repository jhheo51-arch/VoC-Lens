"""상담 파이프라인.

전사본(JSON) → 화자 역할 판정 → 대화 역학 → 발화 감정(텍스트)
  → [음성 파일이 있으면] 음향 특징 → 화자 기준 정규화 → 음성 각성 → 융합
  → 고객 발화의 불만 유형·CX 신호·여정 단계 → 통화 지수·이벤트 → DB

전사본 없이 음성 파일만 있으면 audio.stt로 전사·화자 분리부터 한다 (선택 의존성 필요).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voicecx.audio.apply import Span, apply_audio
from voicecx.audio.dynamics import DynamicsAnalyzer
from voicecx.audio.roles import RoleAssigner
from voicecx.audio.ser import AcousticArousalBaseline
from voicecx.indices.call import score_call
from voicecx.ingest.call_transcript import CallTranscript, load_transcript
from voicecx.schema import Session, SourceType, Unit
from voicecx.store import db
from voicecx.text.cx_signals import CxSignalDetector
from voicecx.text.emotion import LexiconEmotionModel
from voicecx.text.journey import JourneyTagger
from voicecx.text.preprocess import mask_pii, preprocess
from voicecx.text.taxonomy import KeywordTaxonomyClassifier

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac"}
ROLE_LABELS = {"agent": "상담원", "customer": "고객"}
# 고객 발화 직전 무음이 이 길이(초) 이상이면 대기 신호로 본다
WAITING_SILENCE_SEC = 5.0


@dataclass
class CallModels:
    emotion: LexiconEmotionModel
    taxonomy: KeywordTaxonomyClassifier
    cx: CxSignalDetector
    journey: JourneyTagger
    roles: RoleAssigner
    dynamics: DynamicsAnalyzer
    ser: AcousticArousalBaseline

    @classmethod
    def default(cls, settings: dict) -> CallModels:
        taxonomy = KeywordTaxonomyClassifier()
        return cls(
            emotion=LexiconEmotionModel(),
            taxonomy=taxonomy,
            cx=CxSignalDetector(),
            journey=JourneyTagger(taxonomy=taxonomy.taxonomy),
            roles=RoleAssigner(),
            dynamics=DynamicsAnalyzer(settings),
            ser=AcousticArousalBaseline(),
        )

    def versions(self) -> dict[str, str]:
        return {k: getattr(self, k).name for k in ("emotion", "taxonomy", "cx", "journey", "roles", "dynamics", "ser")}


def build_call_session(call: CallTranscript, models: CallModels, settings: dict) -> Session:
    assignment = models.roles.assign(call.turns)
    dynamics = models.dynamics.analyze(call.turns, assignment.roles)
    session_id = f"call:{call.call_id}"

    units = []
    for i, t in enumerate(call.turns):
        confidence = {"role": assignment.confidence}
        if t.stt_conf is not None:
            confidence["stt"] = t.stt_conf
        text = preprocess(t.text).text
        units.append(
            Unit(
                unit_id=f"{session_id}#{i}",
                session_id=session_id,
                source_type=SourceType.CALL,
                order=i,
                text=text,
                clean_text=text,
                speaker_role=assignment.roles.get(t.speaker, "customer"),
                start_sec=t.start,
                end_sec=t.end,
                confidence=confidence,
            )
        )

    models.emotion.predict(units)
    spans = [Span(t.start, t.end, t.text, t.speaker) for t in call.turns]
    audio_meta = apply_audio(units, spans, call.audio_path(), models.ser, settings)

    customer = [u for u in units if u.speaker_role == "customer"]
    models.taxonomy.predict(customer, domain=call.meta.get("domain"))
    models.cx.predict(customer)
    models.journey.predict(customer)
    for unit, flags in zip(units, dynamics.turn_flags):
        if unit.speaker_role != "customer":
            continue
        if flags.repeat_of is not None:
            unit.cx_signals["repeat_contact"] = True
        if flags.silence_before >= WAITING_SILENCE_SEC:
            unit.cx_signals["waiting"] = True

    transcript = " / ".join(f"{ROLE_LABELS.get(u.speaker_role, u.speaker_role)}: {mask_pii(t.text)}" for u, t in zip(units, call.turns))
    meta = {
        **call.meta,
        **dynamics.as_meta(),
        **audio_meta,
        "n_turns": len(units),
        "role_method": assignment.method,
        "role_confidence": assignment.confidence,
        "speaker_roles": assignment.roles,
    }
    session = Session(
        session_id=session_id,
        source_id=call.call_id,
        source_type=SourceType.CALL,
        text=transcript,
        meta=meta,
        units=units,
    )
    return score_call(session, dynamics, settings)


def load_call(path: str | Path, stt_model: str = "medium") -> CallTranscript:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_transcript(path)
    if path.suffix.lower() in AUDIO_SUFFIXES:
        from voicecx.audio.stt import audio_to_transcript

        return CallTranscript.model_validate(audio_to_transcript(path, model_size=stt_model))
    raise ValueError(f"{path}: 전사본(.json) 또는 음성 파일({', '.join(sorted(AUDIO_SUFFIXES))})이어야 합니다")


def expand_paths(paths: list[str | Path]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.suffix.lower() == ".json"))
        else:
            out.append(p)
    return out


def run_call_pipeline(paths: list[str | Path], settings: dict, db_path: str | Path | None = None) -> tuple[str, list[Session]]:
    models = CallModels.default(settings)
    files = expand_paths(paths)
    conn = db.connect(db_path or settings["db_path"])
    run_id = db.start_run(conn, "call", ", ".join(map(str, files)), settings, models.versions())
    try:
        sessions = [build_call_session(load_call(f), models, settings) for f in files]
        tx = models.taxonomy.taxonomy
        db.save_sessions(
            conn, sessions, run_id, label_source="keyword",
            category_of=lambda label: tx.category_of(label) if tx.is_complaint(label) else None,
        )
        db.finish_run(conn, run_id, len(sessions))
    except Exception as exc:
        db.finish_run(conn, run_id, 0, error=repr(exc))
        raise
    finally:
        conn.close()
    return run_id, sessions
