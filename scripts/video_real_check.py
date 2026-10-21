"""실제 YouTube 영상으로 수집·분석이 끝까지 도는지 확인한다 (정답 라벨이 없으므로 정확도가 아니라 동작 점검).

사람이 단 한국어 자막이 있는 공개 영상만 쓴다. 자막·댓글 원문은 저장하지 않고 집계 값만 남긴다.
YOUTUBE_API_KEY(.env)가 있어야 댓글을 받는다. `pip install -e .[video]` 필요.
--stt를 주면 사람 자막이 없는 영상을 Whisper로 전사하고 pyannote로 화자를 나눈 뒤 음향 각성까지 본다
(`pip install -e .[audio]`와 HF_TOKEN 필요). 음성은 임시 폴더에만 두고 분석 후 지운다.

    python scripts/video_real_check.py https://www.youtube.com/watch?v=dStfwAbQB0A https://www.youtube.com/watch?v=-D9zBKMBkeU
    python scripts/video_real_check.py --stt --out video_youtube_stt.json https://www.youtube.com/watch?v=ULoEZw889Lc

결과: experiments/results/video_youtube.json (--out으로 이름 변경)
"""

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter

from voclens.config import REPO_ROOT, load_settings
from voclens.evaluation import save_result
from voclens.ingest.youtube import fetch_bundle
from voclens.pipelines.video import VideoModels, build_video_session

RESULTS = REPO_ROOT / "experiments" / "results"


def summarize(units) -> dict:
    n = len(units)
    return {
        "units": n,
        "neutral_share": round(sum((u.valence or 0) == 0 for u in units) / n, 3),
        "negative_share": round(sum((u.valence or 0) < 0 for u in units) / n, 3),
        "complaint_units": sum(bool(u.complaint_categories) for u in units),
        "top_categories": Counter(c for u in units for c in u.complaint_categories).most_common(5),
        "top_aspects": Counter(u.aspect for u in units if u.aspect).most_common(5),
    }


def audio_summary(bundle, s, speakers, fetch_sec: float) -> dict:
    conf = [seg.stt_conf for seg in bundle.segments if seg.stt_conf is not None]
    audio_a = [u.confidence["audio_arousal"] for u in speakers if "audio_arousal" in u.confidence]
    return {
        "fetch_and_stt_sec": round(fetch_sec, 1),
        "stt_segments": len(bundle.segments),
        "stt_conf_mean": round(statistics.mean(conf), 3) if conf else None,
        "stt_conf_below_0_5": sum(c < 0.5 for c in conf),
        "diarized_speakers": len({seg.speaker for seg in bundle.segments}),
        "guest_share": round(sum(u.speaker_role == "guest" for u in speakers) / len(speakers), 3),
        "audio_used": s.meta.get("audio_used"),
        "audio_note": s.meta.get("audio_note"),
        "snr_db": s.meta.get("snr_db"),
        "audio_arousal_units": len(audio_a),
        "audio_arousal_mean": round(statistics.mean(audio_a), 3) if audio_a else None,
        "audio_arousal_sd": round(statistics.stdev(audio_a), 3) if len(audio_a) > 1 else None,
        "tone_mismatch": s.meta.get("n_tone_mismatch"),
    }


def check(url: str, models: VideoModels, settings: dict, stt: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as audio_dir:
        t0 = time.perf_counter()
        bundle = fetch_bundle(url, allow_stt=stt, audio_dir=audio_dir if stt else None)
        fetch_sec = time.perf_counter() - t0
        s = build_video_session(bundle, models, settings)
    m = s.meta
    speakers = [u for u in s.units if u.speaker_role != "commenter"]
    comments = [u for u in s.units if u.speaker_role == "commenter"]
    labels = [seg["label"] for seg in m["segments"]]
    out = {
        "video_id": bundle.video_id,
        "channel": m.get("channel"),
        "duration_sec": m.get("duration"),
        "views": m.get("views"),
        "transcript_source": bundle.transcript_source,
        "dropped_cues_after_end": m.get("dropped_cues_after_end", 0),
        "sentences": len(speakers),
        "longest_sentence_sec": round(max(u.end_sec - u.start_sec for u in speakers), 1),
        "comments": len(comments),
        "timestamp_comments": m["n_timestamp_comments"],
        "video_type": m["video_type"],
        "segments": len(labels),
        "segment_labels": Counter(labels),
        "segment_label_runs": sum(a != b for a, b in zip(labels, labels[1:])) + 1,
        "engagement": s.indices["engagement"].raw,
        "creator_vs_audience": m["creator_vs_audience"],
        "creator": summarize(speakers),
        "audience": summarize(comments) if comments else None,
    }
    if bundle.transcript_source == "stt":
        out["audio"] = audio_summary(bundle, s, speakers, fetch_sec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--stt", action="store_true", help="사람 자막이 없으면 음성 전사·화자 분리·음향 각성까지")
    ap.add_argument("--out", default="video_youtube.json")
    args = ap.parse_args()
    models, settings = VideoModels.default(), load_settings()
    result = {"note": "정답 라벨 없음. 수집·분석 동작 점검과 분포 확인용", "videos": [check(u, models, settings, args.stt) for u in args.urls]}
    save_result(result, RESULTS / args.out)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
