"""한국어 상담 통화 롤플레이 영상으로 상담 파이프라인(STT → 화자 분리 → 역할 → 음향 각성 → 지수)을 끝까지 돌려 본다.

AI Hub 상담 음성이 승인 전이라, 상담원과 고객 두 사람이 통화하는 공개 롤플레이 영상으로 동작을 점검한다.
정답 전사본·화자 라벨·각성 라벨이 없으므로 정확도가 아니라 동작 점검과 값의 범위 확인이다.

음성(WAV)과 전사 원문은 data/external/call_youtube/에만 둔다 (git 제외. 사람 라벨링 페이지가 이 WAV를 재생한다).
결과 JSON에는 집계 값만 남긴다. `pip install -e .[audio,video]`와 HF_TOKEN 필요.

    python scripts/call_youtube_check.py "https://www.youtube.com/watch?v=h0x8Q3v-51g@30-226" ...
    python scripts/call_youtube_check.py --reuse ...   # 이미 받은 WAV·전사본을 다시 쓴다 (분석만 다시)

결과: experiments/results/call_youtube.json
라벨링: python scripts/make_arousal_labeling_page.py (data/labeling/call_arousal.html)
"""

import argparse
import json
import statistics
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from voclens.audio import acoustic
from voclens.audio.stt import assign_speakers, diarize, transcribe
from voclens.config import REPO_ROOT, load_settings, load_yaml
from voclens.evaluation import save_result
from voclens.ingest.call_transcript import CallTranscript
from voclens.ingest.youtube import _download_wav, video_id_from_url
from voclens.pipelines.call import CallModels, build_call_session

RESULTS = REPO_ROOT / "experiments" / "results"
AUDIO_DIR = REPO_ROOT / "data" / "external" / "call_youtube"
# 말–톤 불일치는 상담 파이프라인에 없어서 영상 파이프라인의 정의(configs/video.yaml)를 그대로 쓴다
MISMATCH = load_yaml("video.yaml")["mismatch"]
CURVE_BINS = 5


def pct(values, q) -> float | None:
    v = [x for x in values if x is not None and not np.isnan(x)]
    return round(float(np.percentile(v, q)), 2) if v else None


def fetch(url: str, clip: tuple[float, float] | None, reuse: bool, model: str) -> tuple[dict, dict]:
    """WAV를 받아 통화 구간만 자르고, 전사·화자 분리를 해서 전사본 dict를 data/external/call_youtube/<id>.json에 남긴다."""
    import yt_dlp

    vid = video_id_from_url(url)
    wav, js = AUDIO_DIR / f"{vid}.wav", AUDIO_DIR / f"{vid}.json"
    if reuse and wav.exists() and js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        return data, data["meta"]["timing"]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timing = {}
    with tempfile.TemporaryDirectory() as tmp:
        base = {"quiet": True, "skip_download": True, "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s")}
        t0 = time.perf_counter()
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=False)
        full = _download_wav(url, base, tmp, vid, Path(tmp) / f"{vid}_full.wav")
        timing["download_sec"] = round(time.perf_counter() - t0, 1)
        signal, sr = acoustic.read_wav(full)
        lo, hi = clip if clip else (0.0, len(signal) / sr)
        with wave.open(str(wav), "wb") as w:   # 통화 부분만 잘라 16kHz 모노 WAV로 남긴다
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(signal[int(lo * sr) : int(hi * sr)], -1, 1) * 32767).astype("<i2").tobytes())
    t0 = time.perf_counter()
    segments = transcribe(wav, model_size=model)
    timing["stt_sec"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    spans = diarize(wav, num_speakers=2)   # 상담 기본값: 두 사람
    timing["diarize_sec"] = round(time.perf_counter() - t0, 1)
    meta = {"title": info.get("title"), "channel": info.get("channel") or info.get("uploader"),
            "duration": round(hi - lo, 1), "video_duration": info.get("duration"), "clip_sec": [lo, hi], "url": f"https://www.youtube.com/watch?v={vid}", "stt_model": model,
            "timing": timing, "diar_spans": [[s.start, s.end, s.speaker] for s in spans]}
    data = {"call_id": vid, "audio": str(wav), "turns": assign_speakers(segments, spans), "meta": meta}
    js.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data, timing


def speech_rates(call: CallTranscript) -> tuple[list[float], list[float], list[bool]]:
    """발화마다 말 속도 두 가지: 규칙이 쓰는 값(한글 음절 / 유성음 프레임 시간)과 참고용(음절 / 발화 길이)."""
    signal, sr = acoustic.read_wav(call.audio_path())
    voiced, plain, reliable = [], [], []
    for t in call.turns:
        f = acoustic.segment_features(signal, sr, t.start, t.end, t.text)
        voiced.append(f.values["speech_rate"])
        plain.append(acoustic._hangul_syllables(t.text) / max(t.end - t.start, 1e-6))
        reliable.append(f.reliable)
    return voiced, plain, reliable


def parse_spec(spec: str) -> tuple[str, tuple[float, float] | None]:
    """"URL@시작-끝"(초) → (URL, 구간). 구간이 없으면 영상 전체."""
    url, _, rng = spec.partition("@")
    if not rng:
        return url, None
    lo, hi = rng.split("-")
    return url, (float(lo), float(hi))


def check(spec: str, models: CallModels, settings: dict, reuse: bool, model: str) -> dict:
    url, clip = parse_spec(spec)
    data, timing = fetch(url, clip, reuse, model)
    meta = data["meta"]
    call = CallTranscript.model_validate({k: data[k] for k in ("call_id", "audio", "turns")} | {"meta": {"domain": None}})
    t0 = time.perf_counter()
    s = build_call_session(call, models, settings)
    timing = {**timing, "pipeline_sec": round(time.perf_counter() - t0, 1)}
    duration = meta["duration"] or call.duration

    # 화자 분리: 화자별 말한 시간 비율 (pyannote 구간 기준)
    talk = Counter()
    for start, end, spk in meta["diar_spans"]:
        talk[spk] += end - start
    total_talk = sum(talk.values()) or 1.0

    units = s.units
    roles = s.meta["speaker_roles"]
    voiced, plain, reliable = speech_rates(call)
    by_role = {}
    for role in ("agent", "customer"):
        idx = [i for i, u in enumerate(units) if u.speaker_role == role]
        audio_a = [units[i].confidence.get("audio_arousal") for i in idx]
        audio_a = [a for a in audio_a if a is not None]
        by_role[role] = {
            "turns": len(idx),
            "speech_rate_voiced_p10_p50_p90": [pct([voiced[i] for i in idx], q) for q in (10, 50, 90)],
            "speech_rate_voiced_p50_reliable_only": pct([voiced[i] for i in idx if reliable[i]], 50),
            "speech_rate_plain_p10_p50_p90": [pct([plain[i] for i in idx], q) for q in (10, 50, 90)],
            "audio_arousal_mean": round(statistics.mean(audio_a), 3) if audio_a else None,
            "audio_arousal_sd": round(statistics.stdev(audio_a), 3) if len(audio_a) > 1 else None,
        }

    # 고객 각성 곡선: 통화를 시간으로 5등분해 구간별 고객 발화의 음향 각성·융합 각성 평균
    curve = []
    for b in range(CURVE_BINS):
        lo, hi = duration * b / CURVE_BINS, duration * (b + 1) / CURVE_BINS
        seg = [u for u in units if u.speaker_role == "customer" and lo <= u.start_sec < hi]
        aa = [u.confidence["audio_arousal"] for u in seg if "audio_arousal" in u.confidence]
        fa = [u.arousal for u in seg if u.arousal is not None]
        curve.append({"bin": b + 1, "turns": len(seg), "audio_arousal": round(statistics.mean(aa), 3) if aa else None,
                      "fused_arousal": round(statistics.mean(fa), 3) if fa else None})

    mismatch = sum(1 for u in units if u.confidence.get("audio_arousal") is not None
                   and (u.valence or 0) >= MISMATCH["text_valence_min"] and u.confidence["audio_arousal"] <= MISMATCH["audio_arousal_max"])
    conf = [t.stt_conf for t in call.turns if t.stt_conf is not None]
    first = call.turns[0].speaker if call.turns else None
    return {
        "video_id": call.call_id,
        "channel": meta["channel"],
        "duration_sec": duration,
        "clip_sec": meta.get("clip_sec"),
        "timing_sec": timing,
        "processing_x_realtime": round((timing["stt_sec"] + timing["diarize_sec"]) / duration, 2) if duration else None,
        "stt_turns": len(call.turns),
        "stt_conf_mean": round(statistics.mean(conf), 3) if conf else None,
        "stt_conf_below_0_5": sum(c < 0.5 for c in conf),
        "diarization": {"speakers": len(talk), "talk_share": {k: round(v / total_talk, 3) for k, v in sorted(talk.items())},
                        "turns_per_speaker": dict(Counter(t.speaker for t in call.turns))},
        "roles": {"method": s.meta["role_method"], "confidence": s.meta["role_confidence"], "speaker_roles": roles,
                  "first_speaker_role": roles.get(first)},
        "audio": {"used": s.meta.get("audio_used"), "snr_db": s.meta.get("snr_db"),
                  "reliable_share": round(sum(reliable) / len(reliable), 3) if reliable else None,
                  "speech_rate_nan_share": round(float(np.mean([np.isnan(v) for v in voiced])), 3) if voiced else None},
        "by_role": by_role,
        "customer_arousal_curve": curve,
        "tone_mismatch_units": mismatch,
        "events": {"count": len(s.events), "up": sum(e.delta > 0 for e in s.events), "down": sum(e.delta < 0 for e in s.events)},
        "indices": {k: {"raw": v.raw, "end": v.end_value} for k, v in s.indices.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+", help="URL 또는 URL@시작-끝(초). 구간은 영상 설명의 챕터 시각(통화 부분)으로 정한다")
    ap.add_argument("--model", default="medium", help="Whisper 모델 크기")
    ap.add_argument("--reuse", action="store_true", help="data/external/call_youtube/의 WAV·전사본을 다시 쓴다")
    ap.add_argument("--out", default="call_youtube.json")
    args = ap.parse_args()
    settings = load_settings()
    models = CallModels.default(settings)
    result = {"note": "정답 라벨 없음. 상담 롤플레이 영상으로 상담 파이프라인 동작 점검. 원문 없음, 집계 값만",
              "ser_model": models.ser.name, "calls": []}
    for url in args.urls:
        result["calls"].append(check(url, models, settings, args.reuse, args.model))
        save_result(result, RESULTS / args.out)   # 영상 하나 끝날 때마다 저장 (오래 걸리므로)
        print(json.dumps(result["calls"][-1], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
