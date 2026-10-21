"""영상 파이프라인.

번들 JSON 또는 URL → 문장 분리(시각 보간) → 주 화자 판정 → 문장 감정(텍스트)
  → [WAV가 있으면] 음향 각성 융합, 말–톤 불일치 표시
  → 불만 유형·속성·CX 신호 → 내용 구간 분할·이름 → 영상 유형
  → 댓글: 감정·불만 유형, 타임스탬프 → 구간 연결
  → 크리에이터 지수(6개), 총평 가중 Valence, 시청자 지수, 구간 요약, 감정 급변 → DB
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voclens.audio.apply import Span, apply_audio
from voclens.audio.ser import AcousticArousalBaseline
from voclens.config import load_yaml
from voclens.indices.compute import score_session
from voclens.indices.events import detect_changes
from voclens.ingest.youtube import VideoBundle, fetch_bundle, load_bundle
from voclens.schema import Event, Session, SourceType, Unit
from voclens.store import db
from voclens.text.cx_signals import CxSignalDetector
from voclens.text.emotion import LexiconEmotionModel
from voclens.text.journey import JourneyTagger
from voclens.text.preprocess import preprocess
from voclens.text.segment import split_sentences
from voclens.text.taxonomy import KeywordTaxonomyClassifier
from voclens.video.classify import VideoTypeClassifier
from voclens.video.comments import compare_creator_audience, extract_timestamps, find_segment
from voclens.video.segment import TextTilingSegmenter

SPEAKER_ROLES = ("creator", "guest")


@dataclass
class VideoModels:
    emotion: LexiconEmotionModel
    taxonomy: KeywordTaxonomyClassifier
    cx: CxSignalDetector
    journey: JourneyTagger
    ser: AcousticArousalBaseline
    video_type: VideoTypeClassifier
    segmenter: TextTilingSegmenter

    @classmethod
    def default(cls) -> VideoModels:
        taxonomy = KeywordTaxonomyClassifier()
        return cls(
            emotion=LexiconEmotionModel(),
            taxonomy=taxonomy,
            cx=CxSignalDetector(),
            journey=JourneyTagger(taxonomy=taxonomy.taxonomy),
            ser=AcousticArousalBaseline(),
            video_type=VideoTypeClassifier(),
            segmenter=TextTilingSegmenter(),
        )

    def versions(self) -> dict[str, str]:
        names = ("emotion", "taxonomy", "cx", "journey", "ser", "video_type", "segmenter")
        return {k: getattr(self, k).name for k in names}


def split_timed(bundle: VideoBundle, boilerplate: re.Pattern | None = None) -> list[Span]:
    """세그먼트를 문장으로 나누고, 세그먼트 안에서 글자 수 비율로 시각을 나눠 준다.

    구독·좋아요 요청 같은 상투 문구 문장은 시각만 차지하고 결과에서는 뺀다.
    """
    spans = []
    for seg in bundle.segments:
        sentences = split_sentences(seg.text) or [seg.text]
        total = sum(len(s) for s in sentences) or 1
        t = seg.start
        for s in sentences:
            dur = (seg.end - seg.start) * len(s) / total
            if not (boilerplate and boilerplate.search(s)):
                spans.append(Span(round(t, 2), round(t + dur, 2), s, seg.speaker or "S0"))
            t += dur
    return spans


def main_speaker(spans: list[Span]) -> str:
    talk: Counter = Counter()
    for s in spans:
        talk[s.speaker] += s.end - s.start
    return talk.most_common(1)[0][0]


def audience_engagement(meta: dict, n_comments: int, n_timestamp_comments: int) -> float | None:
    """영상의 몰입 지수 = 시청자 반응 (좋아요율, 조회 1천 회당 댓글 수, 타임스탬프 댓글 비율).

    리뷰용 몰입 공식(글 길이·속성 수)은 긴 자막에서 항상 최대값이 되어 영상에 쓰지 않는다.
    조회수가 없으면 None을 돌려주고 리뷰 공식을 그대로 둔다.
    좋아요율 5%, 조회 1천 회당 댓글 5개를 각 항목의 최대로 본다 (일반적인 YouTube 반응률보다 높은 기준).
    """
    views = meta.get("views")
    if not views:
        return None
    like_rate = (meta.get("likes") or 0) / views
    comments_per_k = n_comments / (views / 1000)
    ts_share = n_timestamp_comments / n_comments if n_comments else 0.0
    score = 0.5 * min(like_rate / 0.05, 1) + 0.3 * min(comments_per_k / 5, 1) + 0.2 * ts_share
    return round(score * 100, 2)


def _segment_summary(units: list[Unit], comments: list[Unit], segments) -> list[dict]:
    out = []
    for seg in segments:
        members = [u for u in units if u.segment_index == seg.index]
        aspects = Counter(u.aspect for u in members if u.aspect)
        linked = [c for c in comments if c.segment_index == seg.index]
        out.append(
            {
                "index": seg.index,
                "label": seg.label,
                "start_sec": members[0].start_sec,
                "end_sec": members[-1].end_sec,
                "n_sentences": len(members),
                "valence": round(float(np.mean([u.valence or 0.0 for u in members])), 4),
                "aspects": [a for a, _ in aspects.most_common(3)],
                "n_timestamp_comments": len(linked),
                "comment_valence": round(float(np.mean([c.valence or 0.0 for c in linked])), 4) if linked else None,
            }
        )
    return out


def _video_events(session_id: str, units: list[Unit], call_cfg: dict) -> list[Event]:
    events = []
    values = [u.valence or 0.0 for u in units]
    changes = detect_changes(values, window=call_cfg["event_window"], min_delta=call_cfg["event_min_delta"], alpha=call_cfg["event_smoothing"])
    for k, c in enumerate(changes):
        first, last = units[c.start], units[c.end]
        events.append(
            Event(
                event_id=f"{session_id}!valence{k}",
                index_name="valence",
                start_sec=first.start_sec,
                end_sec=last.end_sec,
                delta=c.delta,
                start_unit_id=first.unit_id,
                end_unit_id=last.unit_id,
            )
        )
    return events


def build_video_session(bundle: VideoBundle, models: VideoModels, settings: dict, vcfg: dict | None = None) -> Session:
    vcfg = vcfg or load_yaml("video.yaml")
    session_id = f"video:{bundle.video_id}"
    boilerplate = re.compile("|".join(vcfg["boilerplate"])) if vcfg.get("boilerplate") else None
    spans = split_timed(bundle, boilerplate)
    domain = bundle.meta.get("domain") or vcfg.get("domain")
    creator = main_speaker(spans)

    units = []
    for i, s in enumerate(spans):
        text = preprocess(s.text).text
        units.append(
            Unit(
                unit_id=f"{session_id}#{i}", session_id=session_id, source_type=SourceType.VIDEO, order=i,
                text=text, clean_text=text, speaker_role="creator" if s.speaker == creator else "guest",
                start_sec=s.start, end_sec=s.end,
            )
        )
    models.emotion.predict(units)
    audio_meta = apply_audio(units, spans, bundle.audio_path(), models.ser, settings)

    # 말–톤 불일치: 텍스트는 긍정인데 목소리 각성이 낮음 (음성이 있을 때만).
    # 음성 기준선은 Valence를 내지 않으므로 융합 후 valence는 텍스트 Valence와 같다.
    mm = vcfg["mismatch"]
    n_mismatch = 0
    for u in units:
        audio_a = u.confidence.get("audio_arousal")
        if audio_a is not None and (u.valence or 0) >= mm["text_valence_min"] and audio_a <= mm["audio_arousal_max"]:
            u.cx_signals["tone_mismatch"] = True
            n_mismatch += 1

    models.taxonomy.predict(units, domain=domain)
    models.cx.predict(units)
    models.journey.predict(units)

    segments = models.segmenter.segment([u.text for u in units])
    for seg in segments:
        for u in units[seg.start : seg.end]:
            u.segment, u.segment_index = seg.label, seg.index
    segment_times = [(units[s.start].start_sec, units[s.end - 1].end_sec) for s in segments]

    # 댓글: 발화 뒤에 이어서 저장 (speaker_role = commenter)
    duration = max(s.end for s in spans)
    comments = []
    ccfg = vcfg["comments"]
    for j, c in enumerate(bundle.comments[: ccfg["max_comments"]]):
        text = preprocess(c.text).text
        stamps = extract_timestamps(c.text, max_sec=duration)
        unit = Unit(
            unit_id=f"{session_id}#c{j}", session_id=session_id, source_type=SourceType.VIDEO, order=len(units) + j,
            text=text, clean_text=text, speaker_role="commenter", start_sec=stamps[0] if stamps else None,
            confidence={"likes": float(c.likes)},
        )
        if stamps:
            idx = find_segment(stamps[0], segment_times, ccfg["timestamp_tolerance_sec"])
            if idx is not None:
                unit.segment, unit.segment_index = segments[idx].label, idx
        comments.append(unit)
    if comments:
        models.emotion.predict(comments)
        models.taxonomy.predict(comments, domain=domain)
        models.cx.predict(comments)

    vtype = models.video_type.classify(bundle.meta, [u.text for u in units])
    meta = {
        **{k: v for k, v in bundle.meta.items() if k not in ("description",)},
        "created_at": bundle.meta.get("published_at"),
        "video_type": vtype.type,
        "video_type_label": vtype.label,
        "video_type_confidence": vtype.confidence,
        "transcript_source": bundle.transcript_source,
        "duration": round(duration, 1),
        "n_sentences": len(units),
        "n_comments": len(comments),
        "n_timestamp_comments": sum(c.start_sec is not None for c in comments),
        "n_tone_mismatch": n_mismatch,
        **audio_meta,
    }
    session = Session(
        session_id=session_id, source_id=bundle.video_id, source_type=SourceType.VIDEO,
        text=" ".join(u.text for u in units), meta=meta, units=units + comments,
    )

    idx = settings["indices"]
    speakers = [u for u in units if u.speaker_role in SPEAKER_ROLES]
    score_session(session, idx["above_threshold"], idx["recovery_min_negative"], units=speakers)

    # 총평 구간 가중 Valence: 결론부 평가가 시청자 인식에 큰 영향을 준다
    weights = [vcfg["verdict_weight"] if u.segment == "총평" else 1.0 for u in speakers]
    session.meta["weighted_valence"] = round(float(np.average([u.valence or 0.0 for u in speakers], weights=weights)), 4)
    session.meta["segments"] = _segment_summary(speakers, comments, segments)
    session.meta["creator_vs_audience"] = compare_creator_audience(
        [u.valence or 0.0 for u in speakers], [c.valence or 0.0 for c in comments]
    )
    if comments:
        audience = Session(session_id="tmp", source_id="tmp", source_type=SourceType.VIDEO, units=comments)
        score_session(audience, idx["above_threshold"], idx["recovery_min_negative"])
        # 몰입은 리뷰 공식(글 길이·속성 수)이라 댓글 수백 개를 합치면 항상 100이 된다. 영상 몰입은 audience_engagement로 따로 낸다
        session.meta["audience_indices"] = {k: v.raw for k, v in audience.indices.items() if k not in ("recovery", "engagement")}
    engagement = audience_engagement(bundle.meta, len(comments), session.meta["n_timestamp_comments"])
    if engagement is not None:
        session.indices["engagement"].raw = engagement
        session.meta["engagement_basis"] = "audience"
    session.events = _video_events(session_id, speakers, settings["call"])
    return session


def load_video(source: str, allow_stt: bool = False, audio_dir: str | None = None) -> VideoBundle:
    if source.startswith(("http://", "https://")):
        return fetch_bundle(source, allow_stt=allow_stt, audio_dir=audio_dir)
    return load_bundle(source)


def expand_sources(sources: list[str]) -> list[str]:
    out = []
    for s in sources:
        p = Path(s)
        if not s.startswith("http") and p.is_dir():
            out.extend(str(x) for x in sorted(p.glob("*.json")))
        else:
            out.append(s)
    return out


def run_video_pipeline(sources: list[str], settings: dict, db_path=None, allow_stt: bool = False) -> tuple[str, list[Session]]:
    models = VideoModels.default()
    vcfg = load_yaml("video.yaml")
    items = expand_sources(sources)
    conn = db.connect(db_path or settings["db_path"])
    run_id = db.start_run(conn, "video", ", ".join(items), settings, models.versions())
    try:
        # STT로 받은 음성은 음향 각성을 계산하는 동안만 두고 세션을 만든 뒤 지운다
        with tempfile.TemporaryDirectory() as audio_dir:
            sessions = [build_video_session(load_video(s, allow_stt, audio_dir), models, settings, vcfg) for s in items]
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
