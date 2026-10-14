import numpy as np
import pytest

from voicecx.audio.acoustic import SegmentFeatures, estimate_snr_db, normalize_by_speaker, segment_features
from voicecx.audio.dynamics import DynamicsAnalyzer, char_bigrams, overlap_coefficient
from voicecx.audio.roles import RoleAssigner
from voicecx.audio.ser import AcousticArousalBaseline
from voicecx.audio.stt import SpeakerSpan, SttSegment, assign_speakers
from voicecx.config import load_settings
from voicecx.cx.recovery import categorize
from voicecx.fusion.fuse import fuse_dimension
from voicecx.indices.events import detect_changes, ema
from voicecx.ingest.call_transcript import Turn

SR = 16000


def turn(speaker, start, end, text, role=None):
    return Turn(speaker=speaker, start=start, end=end, text=text, role=role)


def tone(freq, seconds, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- 역할 판정 ---

def test_roles_agent_by_script():
    turns = [
        turn("B", 0, 2, "주문했는데 콜라가 안 왔어요"),
        turn("A", 2.5, 5, "고객님 불편을 드려 죄송합니다 확인해 드리겠습니다"),
        turn("B", 5.5, 7, "언제 와요"),
        turn("A", 7.5, 9, "잠시만 기다려 주세요 고객님"),
    ]
    result = RoleAssigner().assign(turns)
    assert result.roles == {"A": "agent", "B": "customer"}
    assert result.method == "rules" and result.confidence > 0.5


def test_roles_given_and_single_speaker():
    given = RoleAssigner().assign([turn("x", 0, 1, "네", "customer"), turn("y", 1, 2, "네", "agent")])
    assert given.method == "given" and given.roles["y"] == "agent"
    single = RoleAssigner().assign([turn("x", 0, 1, "안녕하세요")])
    assert single.roles == {"x": "customer"}


# --- 대화 역학 ---

@pytest.fixture(scope="module")
def analyzer():
    return DynamicsAnalyzer(load_settings())


ROLES = {"A": "agent", "B": "customer"}


def test_dynamics_interruption_silence_repeat(analyzer):
    turns = [
        turn("A", 0, 3, "무엇을 도와드릴까요"),
        turn("B", 3.5, 7, "한 시간 넘게 기다렸는데 아직 안 왔어요"),
        turn("A", 7.5, 10, "확인해 드리겠습니다"),
        turn("A", 22, 25, "배달 중으로 확인됩니다"),        # 12초 침묵
        turn("B", 24.5, 28, "한 시간 넘게 기다렸다니까요"),  # 끼어들기 + 반복
    ]
    d = analyzer.analyze(turns, ROLES)
    assert d.customer_interruptions == 1
    assert d.max_silence == pytest.approx(12.0)
    assert d.repeat_count == 1
    assert d.turn_flags[4].repeat_of == 1
    assert 0 < d.customer_talk_ratio < 1


@pytest.mark.parametrize(
    "last_agent, outcome",
    [
        ("다시 보내 드리도록 처리했습니다 재발송 완료입니다", "resolved"),
        ("담당 부서로 연결해 드리겠습니다", "transferred"),
        ("확인 후 다시 연락드리겠습니다", "pending"),
        ("네 알겠습니다", "unknown"),
    ],
)
def test_dynamics_outcome(analyzer, last_agent, outcome):
    turns = [turn("A", 0, 1, "무엇을 도와드릴까요"), turn("B", 1, 2, "문의드려요"), turn("A", 2, 3, last_agent)]
    assert analyzer.analyze(turns, ROLES).outcome == outcome


def test_overlap_coefficient():
    a, b = char_bigrams("한 시간 넘게 기다렸어요"), char_bigrams("아니 진짜 한 시간 넘게 기다렸어요 언제 와요")
    assert overlap_coefficient(a, b) == 1.0
    assert overlap_coefficient(a, set()) == 0.0


# --- 음향 특징 ---

def test_pitch_estimate_on_sine():
    signal = tone(150, 1.5)
    f = segment_features(signal, SR, 0, 1.5, "안녕하세요")
    assert f.values["pitch_mean"] == pytest.approx(150, rel=0.03)
    assert f.values["voiced_ratio"] > 0.8
    assert f.reliable


def test_higher_pitch_and_energy_detected():
    low = segment_features(tone(120, 1.0, amp=0.1), SR, 0, 1.0)
    high = segment_features(tone(260, 1.0, amp=0.5), SR, 0, 1.0)
    assert high.values["pitch_mean"] > low.values["pitch_mean"]
    assert high.values["energy_db"] > low.values["energy_db"]


def test_silence_has_no_pitch():
    f = segment_features(np.zeros(SR, dtype=np.float32), SR, 0, 1.0)
    assert np.isnan(f.values["pitch_mean"])
    assert not f.reliable


def test_snr_higher_for_clean_signal():
    rng = np.random.default_rng(0)
    gaps = np.concatenate([tone(150, 1.0), np.zeros(SR // 2, dtype=np.float32)] * 3)
    clean = estimate_snr_db(gaps + rng.normal(0, 0.001, len(gaps)).astype(np.float32), SR)
    noisy = estimate_snr_db(gaps + rng.normal(0, 0.1, len(gaps)).astype(np.float32), SR)
    assert clean > noisy


def _features(pitch, energy):
    values = {k: 0.0 for k in ("energy_std", "pitch_std", "pitch_range", "pause_ratio", "speech_rate")}
    values.update({"pitch_mean": pitch, "energy_db": energy, "voiced_ratio": 0.9})
    return SegmentFeatures(values=values, duration=2.0)


def test_normalize_by_speaker_is_relative_to_each_speaker():
    # 화자 A는 원래 목소리가 높다. A의 평소 피치는 z≈0, B가 높아진 순간만 z가 커야 한다
    feats = [_features(250, -20), _features(252, -20), _features(248, -20),
             _features(120, -25), _features(118, -25), _features(122, -25), _features(180, -15)]
    speakers = ["A", "A", "A", "B", "B", "B", "B"]
    starts = [0, 5, 10, 1, 6, 11, 40]
    zs = normalize_by_speaker(feats, speakers, starts, baseline_sec=30)
    assert abs(zs[0]["pitch_mean"]) < 2
    assert zs[6]["pitch_mean"] > 3
    assert zs[6]["energy_db"] > 3


def test_arousal_baseline_monotonic():
    ser = AcousticArousalBaseline()
    calm = ser.predict_one({"pitch_mean": 0, "energy_db": 0}, reliable=True)
    excited = ser.predict_one({"pitch_mean": 3, "energy_db": 3, "speech_rate": 2}, reliable=True)
    assert excited.arousal > calm.arousal
    assert calm.valence is None
    assert ser.predict_one({"pitch_mean": 3}, reliable=False).confidence < calm.confidence


# --- 융합 ---

def test_fusion_weights_and_missing_modalities():
    both = fuse_dimension(0.2, 1.0, 0.8, 1.0, (0.7, 0.3))
    assert both.value == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)
    text_only = fuse_dimension(0.2, 1.0, None, 1.0, (0.7, 0.3))
    assert text_only.value == 0.2 and text_only.audio_weight == 0
    low_audio_conf = fuse_dimension(0.2, 1.0, 0.8, 0.1, (0.7, 0.3))
    assert low_audio_conf.value < both.value


# --- 이벤트 ---

def test_ema_and_detect_changes():
    assert ema([0, 1], alpha=0.5).tolist() == [0, 0.5]
    values = [-0.5, -0.5, -0.4, 0.5, 0.6, 0.6]
    changes = detect_changes(values, window=2, min_delta=0.35)
    assert len(changes) == 1 and changes[0].delta > 0
    assert detect_changes([0.1, 0.1, 0.1, 0.1]) == []
    assert detect_changes([0.1]) == []


def test_detect_changes_both_directions():
    values = [0.5, 0.5, -0.5, -0.5, -0.5, 0.6, 0.6]
    signs = [np.sign(c.delta) for c in detect_changes(values, window=1, min_delta=0.3, alpha=1.0)]
    assert signs == [-1, 1]


# --- STT 화자 매칭 ---

def test_assign_speakers_by_overlap():
    segments = [SttSegment(0, 2, "안녕하세요", 0.9), SttSegment(2.2, 4, "네 문의드려요", 0.8), SttSegment(10, 11, "끝", 0.7)]
    spans = [SpeakerSpan(0, 2.1, "SPK_0"), SpeakerSpan(2.1, 4.5, "SPK_1")]
    turns = assign_speakers(segments, spans)
    assert [t["speaker"] for t in turns] == ["SPK_0", "SPK_1", "SPK_1"]
    assert turns[0]["stt_conf"] == 0.9


# --- 회복 계기 ---

@pytest.mark.parametrize(
    "text, expected",
    [
        ("불편을 드려 죄송합니다", ["empathy"]),
        ("바로 다시 보내 드리겠습니다", ["resolution"]),
        ("담당 부서로 연결해 드리겠습니다", ["transfer"]),
        ("확인 후 연락드리겠습니다", ["pending"]),
        ("네 알겠습니다", ["other"]),
    ],
)
def test_categorize_agent_phrase(text, expected):
    assert categorize(text) == expected
