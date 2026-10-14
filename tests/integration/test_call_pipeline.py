"""상담 파이프라인 통합 테스트: 전사본(+합성 음성) → DB → SQL → 회복 계기."""

import json
import wave

import numpy as np
import pytest

from voclens.cli import main
from voclens.config import REPO_ROOT, load_settings
from voclens.cx.recovery import recovery_trigger_summary
from voclens.ingest.call_transcript import load_transcript
from voclens.pipelines.call import CallModels, build_call_session, run_call_pipeline
from voclens.store import db

CALLS = REPO_ROOT / "data" / "samples" / "calls"
SR = 16000


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def sessions(settings):
    models = CallModels.default(settings)
    return {p.stem: build_call_session(load_transcript(p), models, settings) for p in sorted(CALLS.glob("*.json"))}


def test_outcomes_match_scenarios(sessions):
    outcomes = {k: s.meta["outcome"] for k, s in sessions.items()}
    assert outcomes == {
        "call_001_missing_item": "resolved",
        "call_002_late_delivery": "pending",
        "call_003_refund_transfer": "transferred",
        "call_004_coupon_question": "resolved",
    }


def test_frustrating_call_scores_highest(sessions):
    frustration = {k: s.indices["frustration"].raw for k, s in sessions.items()}
    assert max(frustration, key=frustration.get) == "call_002_late_delivery"
    peak_end = {k: s.peak_end for k, s in sessions.items()}
    assert max(peak_end, key=peak_end.get) == "call_002_late_delivery"


def test_recovered_call_has_high_recovery_and_trigger(sessions):
    s = sessions["call_001_missing_item"]
    assert s.indices["recovery"].raw > 80
    rises = [e for e in s.events if e.index_name == "valence" and e.delta > 0]
    assert rises
    trigger = next(u for u in s.units if u.unit_id == rises[0].trigger_unit_id)
    assert trigger.speaker_role == "agent"
    assert "다시 보내" in trigger.text


def test_indices_use_customer_turns_only(sessions):
    s = sessions["call_004_coupon_question"]
    assert all(u.complaint_categories == [] for u in s.units if u.speaker_role == "agent")


def _write_tone_wav(path, turns, angry_turns):
    """턴마다 사인파를 넣은 합성 WAV. angry_turns는 피치·음량을 높인다 (실제 음성이 아님)."""
    total = int((max(t["end"] for t in turns) + 0.5) * SR)
    signal = np.zeros(total, dtype=np.float32)
    rng = np.random.default_rng(0)
    for i, t in enumerate(turns):
        start, end = int(t["start"] * SR), int(t["end"] * SR)
        base = 210 if t["speaker"] == "A" else 130
        freq, amp = (base * 1.8, 0.6) if i in angry_turns else (base, 0.15)
        ts = np.arange(end - start) / SR
        signal[start:end] += (amp * np.sin(2 * np.pi * freq * ts)).astype(np.float32)
    signal += rng.normal(0, 0.002, total).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())


def test_audio_path_raises_arousal_for_loud_turn(tmp_path, settings):
    data = json.loads((CALLS / "call_001_missing_item.json").read_text(encoding="utf-8"))
    angry = 4  # "아 진짜 짜증나네요..." 고객 발화
    _write_tone_wav(tmp_path / "call.wav", data["turns"], angry_turns={angry})
    data["audio"] = "call.wav"
    (tmp_path / "call.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    models = CallModels.default(settings)
    with_audio = build_call_session(load_transcript(tmp_path / "call.json"), models, settings)
    without = build_call_session(load_transcript(CALLS / "call_001_missing_item.json"), models, settings)

    assert with_audio.meta["audio_used"] is True and with_audio.meta["snr_db"] > 10
    assert without.meta["audio_used"] is False
    loud = with_audio.units[angry]
    assert loud.confidence["audio_arousal"] > 0.7
    assert loud.arousal > without.units[angry].arousal
    calm_customer = with_audio.units[1]
    assert calm_customer.confidence["audio_arousal"] < loud.confidence["audio_arousal"]


def test_run_pipeline_saves_calls_events_and_queries(tmp_path, settings):
    path = tmp_path / "calls.db"
    run_call_pipeline([CALLS], settings, db_path=path)
    run_call_pipeline([CALLS], settings, db_path=path)  # 재실행해도 중복 없음
    conn = db.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE source_type = 'call'").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] >= 1
    for name in ("call_overview", "call_outcome_summary", "recovery_triggers"):
        assert db.run_query(conn, name)
    summary = {r["category"]: r for r in recovery_trigger_summary(conn)}
    assert summary["해결 조치"]["n_trigger"] >= 1
    assert summary["해결 조치"]["lift"] > 1


def test_cli_call(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "cli.db"), "call", str(CALLS)]) == 0
    out = capsys.readouterr().out
    assert "call_002_late_delivery" in out and "보류" in out


def test_invalid_transcript(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"turns": [{"speaker": "A", "start": 5, "end": 1, "text": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_transcript(bad)
