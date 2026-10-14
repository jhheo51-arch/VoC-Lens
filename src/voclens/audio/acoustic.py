"""발화별 음향 특징 추출 (numpy만 사용하는 기준선).

- 에너지: 프레임 RMS (dB)
- 피치: 프레임 자기상관으로 70~400Hz 기본 주파수 추정
- 휴지 비율: 에너지가 낮은 프레임 비율
- 말 속도: 한글 음절 수 / 유성음 구간 길이

3주차 후반에 openSMILE eGeMAPS로 바꾸거나 함께 쓴다. 출력 키 이름은 유지한다.
특징은 화자마다 높낮이가 달라서 normalize_by_speaker()로 화자 기준 z-score를 만든다.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_SEC = 0.025
HOP_SEC = 0.010
F0_MIN, F0_MAX = 70.0, 400.0
VOICING_THRESHOLD = 0.3
SILENCE_DB_BELOW_PEAK = 35.0

FEATURES = ("energy_db", "energy_std", "pitch_mean", "pitch_std", "pitch_range", "voiced_ratio", "pause_ratio", "speech_rate")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """PCM WAV를 [-1, 1] 범위의 모노 float32로 읽는다. 다른 형식은 ffmpeg로 WAV 변환 후 사용."""
    with wave.open(str(path), "rb") as w:
        sr, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"지원하지 않는 샘플 폭: {width}바이트")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _frames(signal: np.ndarray, sr: int) -> np.ndarray:
    size, hop = int(FRAME_SEC * sr), int(HOP_SEC * sr)
    if len(signal) < size:
        signal = np.pad(signal, (0, size - len(signal)))
    n = 1 + (len(signal) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def _frame_pitch(frame: np.ndarray, sr: int) -> float | None:
    frame = frame - frame.mean()
    energy = float(np.dot(frame, frame))
    if energy <= 1e-10:
        return None
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :] / energy
    lo, hi = int(sr / F0_MAX), min(int(sr / F0_MIN), len(ac) - 1)
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < VOICING_THRESHOLD:
        return None
    # 포물선 보간으로 정수 lag 오차를 줄인다
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag = lag + 0.5 * (a - c) / denom
    return sr / lag


def _hangul_syllables(text: str) -> int:
    return len(re.findall(r"[가-힣]", text))


@dataclass
class SegmentFeatures:
    values: dict[str, float]
    duration: float

    @property
    def reliable(self) -> bool:
        return self.duration >= 1.0 and self.values["voiced_ratio"] >= 0.1


def segment_features(signal: np.ndarray, sr: int, start: float, end: float, text: str = "") -> SegmentFeatures:
    seg = signal[int(start * sr) : int(end * sr)]
    duration = max(len(seg) / sr, 1e-6)
    frames = _frames(seg, sr)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    active = db > (db.max() - SILENCE_DB_BELOW_PEAK)

    pitches = [p for f, a in zip(frames, active) if a and (p := _frame_pitch(f, sr)) is not None]
    pitch = np.asarray(pitches) if pitches else np.asarray([np.nan])
    voiced_ratio = len(pitches) / len(frames)
    voiced_sec = max(len(pitches) * HOP_SEC, 1e-6)

    values = {
        "energy_db": float(db[active].mean()) if active.any() else float(db.mean()),
        "energy_std": float(db[active].std()) if active.any() else 0.0,
        "pitch_mean": float(np.nanmean(pitch)) if pitches else float("nan"),
        "pitch_std": float(np.nanstd(pitch)) if pitches else float("nan"),
        "pitch_range": float(np.nanpercentile(pitch, 90) - np.nanpercentile(pitch, 10)) if pitches else float("nan"),
        "voiced_ratio": float(voiced_ratio),
        "pause_ratio": float(1 - active.mean()),
        "speech_rate": _hangul_syllables(text) / voiced_sec if pitches else float("nan"),
    }
    return SegmentFeatures(values=values, duration=duration)


def estimate_snr_db(signal: np.ndarray, sr: int) -> float:
    """프레임 에너지 분포의 상위 10%와 하위 10% 차이로 대략적인 SNR(dB)을 잰다."""
    db = 20 * np.log10(np.sqrt((_frames(signal, sr) ** 2).mean(axis=1) + 1e-12))
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def normalize_by_speaker(
    features: list[SegmentFeatures], speakers: list[str], starts: list[float], baseline_sec: float = 30.0
) -> list[dict[str, float]]:
    """화자별 z-score. 기준선은 그 화자가 처음 말한 뒤 baseline_sec 안의 발화다.

    기준선 발화가 3개 미만이면 그 화자의 전체 발화를 기준으로 쓴다.
    """
    out: list[dict[str, float]] = [{} for _ in features]
    for speaker in set(speakers):
        idx = [i for i, s in enumerate(speakers) if s == speaker]
        first = starts[idx[0]]
        base = [i for i in idx if starts[i] - first <= baseline_sec and features[i].reliable]
        if len(base) < 3:
            base = [i for i in idx if features[i].reliable] or idx
        for key in FEATURES:
            ref = np.asarray([features[i].values[key] for i in base], dtype=float)
            ref = ref[~np.isnan(ref)]
            mu = float(ref.mean()) if len(ref) else 0.0
            # 표준편차 하한: 기준선이 너무 균일하면 작은 차이도 큰 z가 되는 것을 막는다
            sd = max(float(ref.std()) if len(ref) > 1 else 0.0, abs(mu) * 0.05, 1e-3)
            for i in idx:
                v = features[i].values[key]
                out[i][key] = 0.0 if np.isnan(v) else (v - mu) / sd
    return out
