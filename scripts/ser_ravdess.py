"""음향 각성 규칙(AcousticArousalBaseline)을 공개 감정 음성 RAVDESS로 검증한다.

RAVDESS: 배우 24명(남 12, 여 12)이 영어 문장 2개를 8가지 감정 × 세기 2단계로 읽은 음성 1,440개.
라이선스 CC BY-NC-SA 4.0. 원문 음성은 data/external/ravdess/에 두고 커밋하지 않는다.
  https://zenodo.org/records/1188976  (Audio_Speech_Actors_01-24.zip, 208MB)

배포 규칙의 가중치는 바꾸지 않고 그대로 평가만 한다. 지표는 실행 전에 정했다.
  1. 각성 높음(분노·공포·기쁨·놀람) vs 낮음(중립·평온·슬픔) AUC. 혐오는 각성 분류가 갈려서 뺀다
  2. 같은 감정에서 강하게 읽은 것 vs 보통으로 읽은 것 AUC
  3. 화자 기준선 두 가지
     - 평소 목소리 기준: 그 배우의 중립·평온 음성을 기준선으로 (상담 앞 30초를 기준선으로 쓰는 배포 방식과 같은 구조)
     - 전체 기준: 그 배우의 모든 음성 평균을 기준선으로
     - 정규화 없음: 배우 구분 없이 원래 값 (화자 정규화가 필요한지 확인)
  4. 특징별 AUC (진단용. 규칙을 고치는 데 쓰지 않는다)

한계: 연기한 감정이라 실제 상담보다 과장되어 있다. 문장이 영어라 한글 음절로 세는 말 속도는 0으로 고정된다.

    python scripts/ser_ravdess.py

결과: experiments/results/ser_ravdess.json (음성 없음)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from voclens.audio import acoustic
from voclens.audio.ser import AROUSAL_WEIGHTS, AcousticArousalBaseline
from voclens.config import REPO_ROOT
from voclens.evaluation import save_result

SRC = REPO_ROOT / "data" / "external" / "ravdess"
OUT = REPO_ROOT / "experiments" / "results" / "ser_ravdess.json"

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
HIGH = {"angry", "fearful", "happy", "surprised"}
LOW = {"neutral", "calm", "sad"}
BASELINE_EMOTIONS = {"neutral", "calm"}


def parse(path: Path) -> dict:
    # 03-01-05-02-01-01-12.wav = 음성 전용-말-감정-세기-문장-반복-배우
    _, _, emo, intensity, statement, rep, actor = (int(x) for x in path.stem.split("-"))
    return {"emotion": EMOTIONS[emo], "strong": intensity == 2, "statement": statement, "actor": actor,
            "female": actor % 2 == 0, "path": path}


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    # 양성이 음성보다 클 확률, 동점은 절반 (Mann-Whitney U / n1n2)
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(greater / (len(pos) * len(neg))), 4)


def load(limit_actors: int | None) -> list[dict]:
    files = sorted(SRC.glob("Actor_*/*.wav"))
    rows = [parse(p) for p in files]
    if limit_actors:
        rows = [r for r in rows if r["actor"] <= limit_actors]
    for r in rows:
        signal, sr = acoustic.read_wav(r["path"])
        r["features"] = acoustic.segment_features(signal, sr, 0.0, len(signal) / sr)
        r["snr_db"] = acoustic.estimate_snr_db(signal, sr)
    return rows


def zscores(rows: list[dict], mode: str) -> list[dict]:
    """배포 코드의 normalize_by_speaker를 그대로 쓴다. 기준선은 '처음 말한 뒤 30초 안'이라서
    기준선으로 쓸 음성을 앞(1초 간격)에, 나머지를 뒤(1000초 이후)에 놓아 고른다."""
    if mode == "none":
        speakers = ["all"] * len(rows)
    else:
        speakers = [str(r["actor"]) for r in rows]
    starts = []
    for k, r in enumerate(rows):
        in_base = mode == "all" or (mode == "neutral_calm" and r["emotion"] in BASELINE_EMOTIONS)
        starts.append(k * 0.001 if in_base else 1000.0 + k)
    if mode == "none":
        starts = [0.0] * len(rows)
    return acoustic.normalize_by_speaker([r["features"] for r in rows], speakers, starts, baseline_sec=10_000 if mode != "neutral_calm" else 30.0)


def evaluate(rows: list[dict], arousal: np.ndarray) -> dict:
    high = [a for r, a in zip(rows, arousal) if r["emotion"] in HIGH]
    low = [a for r, a in zip(rows, arousal) if r["emotion"] in LOW]
    strong = [a for r, a in zip(rows, arousal) if r["strong"]]
    normal = [a for r, a in zip(rows, arousal) if not r["strong"] and r["emotion"] != "neutral"]  # 중립은 세기 구분 없음
    out = {
        "auc_high_vs_low": auc(high, low),
        "auc_strong_vs_normal": auc(strong, normal),
        "by_gender": {},
        "mean_arousal_by_emotion": {e: round(float(np.mean([a for r, a in zip(rows, arousal) if r["emotion"] == e])), 3)
                                    for e in EMOTIONS.values()},
    }
    for name, female in (("female", True), ("male", False)):
        h = [a for r, a in zip(rows, arousal) if r["female"] == female and r["emotion"] in HIGH]
        lo = [a for r, a in zip(rows, arousal) if r["female"] == female and r["emotion"] in LOW]
        out["by_gender"][name] = auc(h, lo)
    per_actor = []
    for actor in sorted({r["actor"] for r in rows}):
        h = [a for r, a in zip(rows, arousal) if r["actor"] == actor and r["emotion"] in HIGH]
        lo = [a for r, a in zip(rows, arousal) if r["actor"] == actor and r["emotion"] in LOW]
        per_actor.append(auc(h, lo))
    out["per_actor_auc"] = {"min": min(per_actor), "median": round(float(np.median(per_actor)), 4), "max": max(per_actor),
                            "below_0.7": int(sum(a < 0.7 for a in per_actor)), "actors": len(per_actor)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-actors", type=int, help="빠른 확인용: 배우 1..N만 (이때는 저장하지 않음)")
    args = ap.parse_args()
    rows = load(args.limit_actors)
    ser = AcousticArousalBaseline()

    result = {
        "data": {"clips": len(rows), "actors": len({r["actor"] for r in rows}),
                 "high_arousal": sum(r["emotion"] in HIGH for r in rows), "low_arousal": sum(r["emotion"] in LOW for r in rows),
                 "excluded_disgust": sum(r["emotion"] == "disgust" for r in rows),
                 "reliable_share": round(float(np.mean([r["features"].reliable for r in rows])), 3)},
        "model": {"name": ser.name, "weights": AROUSAL_WEIGHTS, "note": "가중치를 바꾸지 않고 평가만 함. 말 속도는 영어라 0으로 고정"},
        "normalization": {},
        "feature_auc_high_vs_low": {},
    }
    for mode in ("neutral_calm", "all", "none"):
        zs = zscores(rows, mode)
        preds = ser.predict(zs, [r["features"].reliable for r in rows], snr_db=None)
        arousal = np.array([p.arousal for p in preds], dtype=float)
        result["normalization"][mode] = evaluate(rows, arousal)
        if mode == "neutral_calm":
            for key in acoustic.FEATURES:
                v = [z[key] for z in zs]
                result["feature_auc_high_vs_low"][key] = auc([x for r, x in zip(rows, v) if r["emotion"] in HIGH],
                                                             [x for r, x in zip(rows, v) if r["emotion"] in LOW])
    if not args.limit_actors:
        save_result(result, OUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
