"""영상 말 감성(감정 사전 vs 학습 감정 모델)과 내용 구간 분할(v0.1 texttiling vs v0.2 position)을 같은 영상에서 비교한다.

정답 라벨이 없으므로 정확도가 아니라 아래의 미리 정한 구조 점검 지표로 본다 (결과를 보기 전에 정함, 2026-10-21).

말 감성 (크리에이터 문장, 댓글 따로)
  - zero_share: Valence가 정확히 0인 비율 (9절의 '감성 0')
  - uncalled_share: 감정 표현을 잡지 못한 비율. 사전은 걸린 표현 0개, 모델은 6감정 확률이 모두 0.5 미만 (둘 다 text 신뢰도 0.2)
  - 분포: V < −0.2(분류 불가 기준 아래) / −0.2 ≤ V < 0 / V = 0 / 0 < V < 0.25 / V ≥ 0.25(칭찬 기준), 평균·표준편차
  - sign_agreement: 두 방식이 모두 감정을 잡은 문장에서 Valence 부호가 같은 비율
구간 분할 (위치는 구간 시각 ÷ 영상 길이)
  - n_segments: 구간 수
  - intro_outside_front20: 시작이 앞 20% 밖인 도입 구간 수
  - verdict_count, verdict_outside_tail30: 총평 구간 수, 가운데가 뒤 30% 밖인 총평 구간 수
  - has_verdict_in_tail30: 뒤 30% 안에 총평 구간이 있는가
  - misfit_labels: 제품군(제목·태그)에 맞지 않는 구간 이름 수 (자동차의 언박싱 등)
  - default_share: 이름을 못 붙여 '본문'이 된 문장 비율
  - chapters: 설명란에 챕터 시각이 3개 이상이면 챕터 경계와 비교. 예측 경계 ±15초 안에 챕터 경계가 있는 비율(precision),
    챕터 경계 ±15초 안에 예측 경계가 있는 비율(recall). 챕터는 크리에이터가 단 목차라 정답은 아니고 참고 기준이다.
주의: intro_outside_front20·verdict_outside_tail30·misfit_labels는 v0.2 규칙이 직접 막는 항목이라 v0.2에서 0이 되는 것은
규칙이 의도대로 적용됐다는 확인일 뿐이다. 개선 여부는 구간 수, 총평 유무, 본문 비율, 챕터 일치로 본다.

번들은 scripts/video_bundle_cache.py로 data/external/youtube/ 에 받아 둔 것을 쓴다. 원문은 결과에 넣지 않는다.

    python scripts/video_speech_segment_check.py --split dev --out video_speech_segment_dev.json -- dStfwAbQB0A -D9zBKMBkeU ULoEZw889Lc
    python scripts/video_speech_segment_check.py --split eval --out video_speech_segment_eval.json -- crOa9gsD9tE hNwemeKKNGs AWmjVAoC_zE

결과: experiments/results/video_speech_segment_{dev,eval}.json
"""

import argparse
import json
import re
import statistics
from collections import Counter

from voclens.config import REPO_ROOT, load_settings, load_yaml
from voclens.evaluation import save_result
from voclens.ingest.youtube import load_bundle
from voclens.pipelines.video import VideoModels, build_video_session
from voclens.video.segment import product_group

RESULTS = REPO_ROOT / "experiments" / "results"
CACHE = REPO_ROOT / "data" / "external" / "youtube"
EMOTION_MODEL = REPO_ROOT / "models" / "emotion-kote-small"
CHAPTER = re.compile(r"(?m)^\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{2})\)?\s")
TOLERANCE_SEC = 15
UNCALLED_CONF = 0.2


def _sec(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    return sum(p * 60**i for i, p in enumerate(reversed(parts)))


def chapter_starts(description: str, duration: float) -> list[int]:
    stamps = sorted({_sec(m) for m in CHAPTER.findall(description or "")})
    return [s for s in stamps if 0 < s < duration]


def emotion_summary(units) -> dict:
    v = [u.valence or 0.0 for u in units]
    n = len(v)
    share = lambda f: round(sum(f(x) for x in v) / n, 3)   # noqa: E731
    return {
        "units": n,
        "zero_share": share(lambda x: x == 0),
        "uncalled_share": round(sum(u.confidence.get("text", 0) <= UNCALLED_CONF for u in units) / n, 3),
        "dist": {
            "neg_below_-0.2": share(lambda x: x < -0.2),
            "neg_-0.2_0": share(lambda x: -0.2 <= x < 0),
            "zero": share(lambda x: x == 0),
            "pos_0_0.25": share(lambda x: 0 < x < 0.25),
            "pos_0.25_up": share(lambda x: x >= 0.25),
        },
        "valence_mean": round(statistics.mean(v), 4),
        "valence_sd": round(statistics.stdev(v), 4) if n > 1 else None,
    }


def sign_agreement(a_units, b_units) -> dict:
    pairs = [
        (a.valence or 0, b.valence or 0) for a, b in zip(a_units, b_units)
        if a.confidence.get("text", 0) > UNCALLED_CONF and b.confidence.get("text", 0) > UNCALLED_CONF and (a.valence or 0) and (b.valence or 0)
    ]
    return {"both_called": len(pairs), "same_sign": round(sum((x > 0) == (y > 0) for x, y in pairs) / len(pairs), 3) if pairs else None}


def segment_summary(session, groups: dict, description: str = "") -> dict:
    """description은 번들의 설명란 (세션 meta에는 설명란을 넣지 않는다)."""
    m = session.meta
    dur = m["duration"]
    segs = m["segments"]
    speakers = [u for u in session.units if u.speaker_role != "commenter"]
    group = product_group(m, groups)
    excluded = set(groups[group].get("exclude_labels", [])) if group else set()
    pos = [(s["label"], s["start_sec"] / dur, (s["start_sec"] + s["end_sec"]) / 2 / dur) for s in segs]
    out = {
        "n_segments": len(segs),
        "labels": [s["label"] for s in segs],
        "label_counts": Counter(s["label"] for s in segs),
        "intro_count": sum(l == "도입" for l, _, _ in pos),
        "intro_outside_front20": sum(l == "도입" and st > 0.2 for l, st, _ in pos),
        "verdict_count": sum(l == "총평" for l, _, _ in pos),
        "verdict_outside_tail30": sum(l == "총평" and mid < 0.7 for l, _, mid in pos),
        "has_verdict_in_tail30": any(l == "총평" and mid >= 0.7 for l, _, mid in pos),
        "product_group": group,
        "misfit_labels": sum(l in excluded for l, _, _ in pos),
        "default_share": round(sum(u.segment == "본문" for u in speakers) / len(speakers), 3),
    }
    chapters = chapter_starts(description, dur)
    if len(chapters) >= 3:
        pred = [s["start_sec"] for s in segs[1:]]
        hit = lambda t, ref: any(abs(t - r) <= TOLERANCE_SEC for r in ref)   # noqa: E731
        out["chapters"] = {
            "n_chapter_boundaries": len(chapters),
            "n_predicted_boundaries": len(pred),
            "precision": round(sum(hit(t, chapters) for t in pred) / len(pred), 3) if pred else None,
            "recall": round(sum(hit(c, pred) for c in chapters) / len(chapters), 3),
        }
    return out


def check(video_id: str, before: VideoModels, after: VideoModels, settings: dict, groups: dict) -> dict:
    bundle = load_bundle(CACHE / f"{video_id}.json")
    a = build_video_session(bundle, before, settings)
    b = build_video_session(load_bundle(CACHE / f"{video_id}.json"), after, settings)
    role = lambda s, r: [u for u in s.units if (u.speaker_role == "commenter") == r]   # noqa: E731
    return {
        "video_id": video_id,
        "channel": a.meta.get("channel"),
        "duration_sec": a.meta["duration"],
        "transcript_source": bundle.transcript_source,
        "sentences": len(role(a, False)),
        "comments": len(role(a, True)),
        "emotion": {
            "creator": {"lexicon": emotion_summary(role(a, False)), "model": emotion_summary(role(b, False)),
                        "sign_agreement": sign_agreement(role(a, False), role(b, False))},
            "audience": ({"lexicon": emotion_summary(role(a, True)), "model": emotion_summary(role(b, True)),
                          "sign_agreement": sign_agreement(role(a, True), role(b, True))} if role(a, True) else None),
            "creator_vs_audience": {"lexicon": a.meta["creator_vs_audience"], "model": b.meta["creator_vs_audience"]},
        },
        "segmentation": {"texttiling_v0.1": segment_summary(a, groups, bundle.meta.get("description", "")),
                         "position_v0.2": segment_summary(b, groups, bundle.meta.get("description", ""))},
    }


def totals(videos: list[dict]) -> dict:
    out = {}
    for key in ("texttiling_v0.1", "position_v0.2"):
        s = [v["segmentation"][key] for v in videos]
        out[key] = {
            "n_segments": sum(x["n_segments"] for x in s),
            "intro_outside_front20": sum(x["intro_outside_front20"] for x in s),
            "verdict_outside_tail30": sum(x["verdict_outside_tail30"] for x in s),
            "videos_with_verdict_in_tail30": sum(x["has_verdict_in_tail30"] for x in s),
            "misfit_labels": sum(x["misfit_labels"] for x in s),
            "default_share_mean": round(statistics.mean(x["default_share"] for x in s), 3),
        }
    for key in ("lexicon", "model"):
        c = [v["emotion"]["creator"][key] for v in videos]
        n = sum(x["units"] for x in c)
        out[f"creator_{key}"] = {
            "units": n,
            "zero_share": round(sum(x["zero_share"] * x["units"] for x in c) / n, 3),
            "uncalled_share": round(sum(x["uncalled_share"] * x["units"] for x in c) / n, 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+", help="data/external/youtube/<id>.json 으로 받아 둔 영상 ID")
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--emotion-model", default=str(EMOTION_MODEL))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    settings = load_settings()
    before = VideoModels.default(segmenter="texttiling", settings=settings)
    after = VideoModels.default(emotion_model=args.emotion_model, segmenter="position", settings=settings)
    groups = load_yaml("video.yaml")["position_segmentation"]["product_groups"]
    videos = [check(v, before, after, settings, groups) for v in args.video_ids]
    result = {
        "note": "정답 라벨 없음. 미리 정한 구조 점검 지표로 변경 전후 비교 (원문 미포함)",
        "split": args.split,
        "before": before.versions(),
        "after": after.versions(),
        "totals": totals(videos),
        "videos": videos,
    }
    save_result(result, RESULTS / args.out)
    print(json.dumps(result["totals"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
