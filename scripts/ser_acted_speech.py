"""음향 각성 규칙(AcousticArousalBaseline)을 공개 감정 연기 음성으로 검증한다.

데이터 (원문 음성은 data/external/에 두고 커밋하지 않는다)
  ravdess  배우 24명(남 12, 여 12), 영어 문장 2개 × 8감정 × 세기 2단계, 1,440개. CC BY-NC-SA 4.0
           https://zenodo.org/records/1188976  (Audio_Speech_Actors_01-24.zip) → data/external/ravdess/Actor_*/
  crema_d  배우 91명(남 48, 여 43), 영어 문장 12개 × 6감정, 7,442개. ODbL 1.0
           https://github.com/CheyneyComputerScience/CREMA-D (AudioWAV/, VideoDemographics.csv) → data/external/crema_d/

배포 규칙의 가중치는 바꾸지 않고 그대로 평가만 한다. 지표는 실행 전에 정했다.
  1. 각성 높음 vs 낮음 AUC. 혐오는 각성 분류가 갈려서 뺀다
     ravdess: 분노·공포·기쁨·놀람 vs 중립·평온·슬픔 / crema_d: 분노·공포·기쁨 vs 중립·슬픔
  2. 강하게 읽은 것 vs 약하게 읽은 것 AUC
     ravdess: 세기 2 vs 1 (중립 제외) / crema_d: 세기가 정해진 문장(IEO)의 HI vs LO
  3. 화자 기준선
     - baseline_neutral: 그 배우의 중립(·평온) 음성을 기준선으로 (상담 앞 30초를 기준선으로 쓰는 배포 방식과 같은 구조)
     - all: 그 배우의 모든 음성 평균을 기준선으로
     - none: 배우 구분 없이 원래 값 (화자 정규화가 필요한지 확인)
  4. 특징별 AUC (진단용. 규칙을 고치는 데 쓰지 않는다). 에너지 하나만 쓴 것과의 비교는 crema_d 실행 전에 정했다

한계: 연기한 감정이라 실제 상담보다 과장되어 있다. 문장이 영어라 한글 음절로 세는 말 속도는 0으로 고정된다.

    python scripts/ser_acted_speech.py ravdess
    python scripts/ser_acted_speech.py crema_d

결과: experiments/results/ser_<데이터>.json (음성 없음)
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voclens.audio import acoustic
from voclens.audio.ser import AROUSAL_WEIGHTS, AcousticArousalBaseline
from voclens.config import REPO_ROOT
from voclens.evaluation import save_result

EXTERNAL = REPO_ROOT / "data" / "external"
RESULTS = REPO_ROOT / "experiments" / "results"


@dataclass(frozen=True)
class Dataset:
    emotions: tuple[str, ...]
    high: frozenset[str]
    low: frozenset[str]
    baseline: frozenset[str]


DATASETS = {
    "ravdess": Dataset(("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"),
                       frozenset({"angry", "fearful", "happy", "surprised"}), frozenset({"neutral", "calm", "sad"}),
                       frozenset({"neutral", "calm"})),
    "crema_d": Dataset(("neutral", "happy", "sad", "angry", "fearful", "disgust"),
                       frozenset({"angry", "fearful", "happy"}), frozenset({"neutral", "sad"}), frozenset({"neutral"})),
}
CREMA_EMOTIONS = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}


def list_ravdess() -> list[dict]:
    rows = []
    for path in sorted((EXTERNAL / "ravdess").glob("Actor_*/*.wav")):
        # 03-01-05-02-01-01-12.wav = 음성 전용-말-감정-세기-문장-반복-배우
        _, _, emo, intensity, _, _, actor = (int(x) for x in path.stem.split("-"))
        emotion = DATASETS["ravdess"].emotions[emo - 1]
        strong = None if emotion == "neutral" else intensity == 2   # 중립은 세기 구분 없음
        rows.append({"emotion": emotion, "strong": strong, "actor": str(actor), "female": actor % 2 == 0, "path": path})
    return rows


def list_crema_d() -> list[dict]:
    root = EXTERNAL / "crema_d"
    with open(root / "VideoDemographics.csv", encoding="utf-8") as f:
        sex = {r["ActorID"]: r["Sex"] for r in csv.DictReader(f)}
    rows = []
    for path in sorted((root / "AudioWAV").glob("*.wav")):
        # 1001_IEO_ANG_HI.wav = 배우_문장_감정_세기 (세기는 IEO 문장에만 있고 나머지는 XX)
        actor, _, emo, level = path.stem.split("_")
        strong = {"HI": True, "LO": False}.get(level)
        rows.append({"emotion": CREMA_EMOTIONS[emo], "strong": strong, "actor": actor, "female": sex[actor] == "Female",
                     "path": path})
    return rows


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    # 양성이 음성보다 클 확률, 동점은 절반 (Mann-Whitney U / n1n2)
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(greater / (len(pos) * len(neg))), 4)


def load(rows: list[dict], limit_actors: int | None) -> tuple[list[dict], list[str]]:
    if limit_actors:
        keep = sorted({r["actor"] for r in rows})[:limit_actors]
        rows = [r for r in rows if r["actor"] in keep]
    loaded, broken = [], []
    for r in rows:
        try:
            signal, sr = acoustic.read_wav(r["path"])
        except Exception:  # noqa: BLE001  깨진 파일은 세어서 기록만 한다
            broken.append(r["path"].name)
            continue
        if len(signal) < sr * 0.1:
            broken.append(r["path"].name)
            continue
        r["features"] = acoustic.segment_features(signal, sr, 0.0, len(signal) / sr)
        loaded.append(r)
    return loaded, broken


def zscores(rows: list[dict], mode: str, ds: Dataset) -> list[dict]:
    """배포 코드의 normalize_by_speaker를 그대로 쓴다. 기준선은 '처음 말한 뒤 30초 안'이라서
    기준선으로 쓸 음성을 앞(1ms 간격)에, 나머지를 뒤(1000초 이후)에 놓아 고른다."""
    feats = [r["features"] for r in rows]
    if mode == "none":
        return acoustic.normalize_by_speaker(feats, ["all"] * len(rows), [0.0] * len(rows), baseline_sec=10_000)
    speakers = [r["actor"] for r in rows]
    if mode == "all":
        return acoustic.normalize_by_speaker(feats, speakers, [k * 0.001 for k in range(len(rows))], baseline_sec=10_000)
    starts = [k * 0.001 if r["emotion"] in ds.baseline else 1000.0 + k for k, r in enumerate(rows)]
    return acoustic.normalize_by_speaker(feats, speakers, starts, baseline_sec=30.0)


def high_low_auc(pairs, ds: Dataset) -> float:
    return auc([a for r, a in pairs if r["emotion"] in ds.high], [a for r, a in pairs if r["emotion"] in ds.low])


def evaluate(rows: list[dict], arousal: np.ndarray, ds: Dataset) -> dict:
    pairs = list(zip(rows, arousal))
    out = {
        "auc_high_vs_low": high_low_auc(pairs, ds),
        "auc_strong_vs_normal": auc([a for r, a in pairs if r["strong"] is True], [a for r, a in pairs if r["strong"] is False]),
        "by_gender": {name: high_low_auc([(r, a) for r, a in pairs if r["female"] == female], ds)
                      for name, female in (("female", True), ("male", False))},
        "mean_arousal_by_emotion": {e: round(float(np.mean([a for r, a in pairs if r["emotion"] == e])), 3) for e in ds.emotions},
    }
    per_actor = [high_low_auc([(r, a) for r, a in pairs if r["actor"] == actor], ds) for actor in sorted({r["actor"] for r in rows})]
    out["per_actor_auc"] = {"min": min(per_actor), "median": round(float(np.median(per_actor)), 4), "max": max(per_actor),
                            "below_0.7": int(sum(a < 0.7 for a in per_actor)), "actors": len(per_actor)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--limit-actors", type=int, help="빠른 확인용: 앞 배우 N명만 (이때는 저장하지 않음)")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    listed = list_ravdess() if args.dataset == "ravdess" else list_crema_d()
    rows, broken = load(listed, args.limit_actors)
    ser = AcousticArousalBaseline()

    result = {
        "data": {"clips": len(rows), "broken_files": broken, "actors": len({r["actor"] for r in rows}),
                 "female_actors": len({r["actor"] for r in rows if r["female"]}),
                 "high_arousal": sum(r["emotion"] in ds.high for r in rows), "low_arousal": sum(r["emotion"] in ds.low for r in rows),
                 "excluded_disgust": sum(r["emotion"] == "disgust" for r in rows),
                 "strong": sum(r["strong"] is True for r in rows), "normal": sum(r["strong"] is False for r in rows),
                 "reliable_share": round(float(np.mean([r["features"].reliable for r in rows])), 3)},
        "model": {"name": ser.name, "weights": AROUSAL_WEIGHTS, "note": "가중치를 바꾸지 않고 평가만 함. 말 속도는 영어라 0으로 고정"},
        "normalization": {},
        "feature_auc_high_vs_low": {},
    }
    for mode in ("baseline_neutral", "all", "none"):
        zs = zscores(rows, mode, ds)
        preds = ser.predict(zs, [r["features"].reliable for r in rows], snr_db=None)
        result["normalization"][mode] = evaluate(rows, np.array([p.arousal for p in preds], dtype=float), ds)
        if mode == "baseline_neutral":
            for key in acoustic.FEATURES:
                result["feature_auc_high_vs_low"][key] = high_low_auc(list(zip(rows, [z[key] for z in zs])), ds)
    if not args.limit_actors:
        save_result(result, RESULTS / f"ser_{args.dataset}.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
