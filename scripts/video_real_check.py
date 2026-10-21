"""실제 YouTube 영상으로 수집·분석이 끝까지 도는지 확인한다 (정답 라벨이 없으므로 정확도가 아니라 동작 점검).

사람이 단 한국어 자막이 있는 공개 영상만 쓴다. 자막·댓글 원문은 저장하지 않고 집계 값만 남긴다.
YOUTUBE_API_KEY(.env)가 있어야 댓글을 받는다. `pip install -e .[video]` 필요.

    python scripts/video_real_check.py https://www.youtube.com/watch?v=dStfwAbQB0A https://www.youtube.com/watch?v=-D9zBKMBkeU

결과: experiments/results/video_youtube.json
"""

import argparse
import json
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


def check(url: str, models: VideoModels, settings: dict) -> dict:
    bundle = fetch_bundle(url)
    s = build_video_session(bundle, models, settings)
    m = s.meta
    speakers = [u for u in s.units if u.speaker_role != "commenter"]
    comments = [u for u in s.units if u.speaker_role == "commenter"]
    labels = [seg["label"] for seg in m["segments"]]
    return {
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
        "audience": summarize(comments),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    args = ap.parse_args()
    models, settings = VideoModels.default(), load_settings()
    result = {"note": "정답 라벨 없음. 수집·분석 동작 점검과 분포 확인용", "videos": [check(u, models, settings) for u in args.urls]}
    save_result(result, RESULTS / "video_youtube.json")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
