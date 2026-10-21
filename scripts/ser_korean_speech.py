"""음향 각성 규칙(AcousticArousalBaseline)을 공개 한국어 감정 음성 두 가지로 검증한다 (8절의 한국어판).

8절은 영어 연기 음성이라 한글 음절로 세는 말 속도(가중치 0.20)가 0으로 고정됐다. 여기서는 한국어 문장을
함께 넘겨 말 속도를 처음으로 잰다. 로그인·신청 없이 받을 수 있고 라이선스가 적힌 데이터만 쓴다.

데이터 (원문 음성은 data/external/에 두고 커밋하지 않는다)
  hi_kia  Hi,KIA (KAIST, APSIPA 2022). 배우 8명(여 4, 남 4)이 호출어 "하이 기아"를 4감정으로 읽은 488개. CC BY 4.0
          https://zenodo.org/records/7091465 (hi_kia_1.0.tar.gz) → data/external/hi_kia/
  deeply  Deeply Korean Read Speech Corpus 공개 표본 (Deeply Inc., OpenSLR 97). 화자 2명(여 20세, 49세)이 대화 대본을
          목소리 감정 3가지(부정·중립·긍정)로 읽은 1,697개. 방 3곳(무반향실·원룸·댄스 연습실). CC BY-NC-ND 4.0
          https://www.openslr.org/97/ (KoreanReadSpeechCorpus.tar.gz) → data/external/deeply_krsc/
          메타데이터: github.com/deeplyinc/Korean-Read-Speech-Corpus 의 dataset/Korean_Read_Speech_Corpus_sample.json

배포 규칙의 가중치는 바꾸지 않고 평가만 한다. 지표는 실행 전에 정했다.
  hi_kia  1. 각성 높음(분노·기쁨) vs 낮음(중립·슬픔) AUC, 배우별 최저 AUC, 성별 AUC (8절과 같음)
          2. 기준선: 그 배우의 중립 음성 / 그 배우의 모든 음성 / 정규화 없음
          문장이 모두 "하이 기아"(4음절)라 말 속도 = 4 / 유성음 시간. 말 속도 차이는 순수하게 읽는 빠르기다
  deeply  감정 라벨이 각성이 아니라 목소리의 긍정·부정(valence)이다. 그래서 미리 아래처럼 정했다
          1. 감정을 실은 목소리(부정·긍정) vs 중립 목소리 AUC. 부정 vs 중립, 긍정 vs 중립도 따로
          2. 긍정 vs 부정 AUC: 각성 규칙은 긍정·부정을 가르지 못해야 정상(0.5 근처)이다
          3. 화자 기준선 단위는 화자 × 방 (상담 한 통화 = 한 화자, 한 환경과 같은 구조). 방별 AUC로 울림의 영향을 본다
  공통    특징별 AUC와, 말 속도를 0으로 둔 규칙(=8절 영어 조건)과의 비교 (진단용. 규칙을 고치는 데 쓰지 않는다)
          말 속도 값의 범위(음절/초)를 그대로 기록해 한국어에서 정상 범위인지 본다

한계: 둘 다 낭독·연기 음성이다. hi_kia는 1초 안팎의 호출어라 배포 코드의 '신뢰할 수 있는 발화'(1초 이상) 기준에 걸리는 것이 많다.
deeply는 화자 2명뿐이라 화자별·성별 결과는 낼 수 없다.

    python scripts/ser_korean_speech.py hi_kia
    python scripts/ser_korean_speech.py deeply

결과: experiments/results/ser_<데이터>.json (음성·문장 없음)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ser_acted_speech import auc   # 같은 AUC 정의(동점 절반)를 쓴다
from voclens.audio import acoustic
from voclens.audio.ser import AROUSAL_WEIGHTS, AcousticArousalBaseline
from voclens.config import REPO_ROOT
from voclens.evaluation import save_result

EXTERNAL = REPO_ROOT / "data" / "external"
RESULTS = REPO_ROOT / "experiments" / "results"
HI_KIA_TEXT = "하이 기아"
HI_KIA_EMOTIONS = {"a": "angry", "h": "happy", "s": "sad", "n": "neutral"}
VOICE = {-1: "negative", 0: "neutral", 1: "positive"}
ROOMS = {"AirbnbStudio": "studio_apartment", "AnechoicChamber": "anechoic", "DanceStudio": "dance_studio"}


def list_hi_kia() -> list[dict]:
    rows = []
    for path in sorted((EXTERNAL / "hi_kia").rglob("*.wav")):
        # F5_S01_1_a.wav = 배우(성별+번호)_장면_시도_감정 (압축을 풀어 확인한 실제 이름)
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[-1][:1].lower() not in HI_KIA_EMOTIONS:
            continue
        actor, _, _, emo = parts
        rows.append({"label": HI_KIA_EMOTIONS[emo[:1].lower()], "group": actor, "female": actor[:1].upper() == "F",
                     "room": None, "text": HI_KIA_TEXT, "path": path, "fname": path.stem})
    return rows


def hi_kia_human_agreement() -> dict[str, bool]:
    """annotation/human.csv: 청취자 8명(V1~V8)이 고른 감정. 다수결이 배우의 의도 감정과 같은 파일인지.

    human.csv의 코드는 a=분노, e=기쁨, d=슬픔, n=중립이다 (의도 감정별 개수가 파일 이름의 a/h/s/n 개수와 같다).
    """
    import csv
    from collections import Counter

    path = next((EXTERNAL / "hi_kia").rglob("human.csv"), None)
    if path is None:
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            votes = Counter(row[f"V{k}"] for k in range(1, 9))
            top, n = votes.most_common(1)[0]
            out[row["fname"][:-1] + {"a": "a", "e": "h", "d": "s", "n": "n"}[row["EMOTION"]]] = top == row["EMOTION"] and n >= 5
    return out


def list_deeply() -> list[dict]:
    root = EXTERNAL / "deeply_krsc"
    meta = json.loads((root / "Korean_Read_Speech_Corpus_sample.json").read_text(encoding="utf-8"))
    rows = []
    for room, items in meta.items():
        for key, m in items.items():
            path = root / room / f"{key}.wav"
            if not path.exists():
                continue
            rows.append({"label": VOICE[m["voice_sentiment"]], "text_sentiment": m["text_sentiment"],
                         "group": f"{m['speaker']}@{ROOMS[room]}", "speaker": m["speaker"], "female": m["sex"] == 0,
                         "room": ROOMS[room], "text": m["text"], "path": path})
    return rows


DATASETS = {
    # high/low: 주 지표의 양성·음성, baseline: 화자 기준선으로 쓸 라벨
    "hi_kia": {"list": list_hi_kia, "high": {"angry", "happy"}, "low": {"neutral", "sad"}, "baseline": {"neutral"},
               "labels": ("neutral", "sad", "happy", "angry")},
    "deeply": {"list": list_deeply, "high": {"negative", "positive"}, "low": {"neutral"}, "baseline": {"neutral"},
               "labels": ("neutral", "negative", "positive")},
}


def load(rows: list[dict]) -> tuple[list[dict], list[str]]:
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
        # 8절과 달리 문장을 넘긴다 → 말 속도(한글 음절 / 유성음 시간)가 계산된다
        r["features"] = acoustic.segment_features(signal, sr, 0.0, len(signal) / sr, r["text"])
        r["plain_rate"] = len(re.findall(r"[가-힣]", r["text"])) / (len(signal) / sr)
        loaded.append(r)
    return loaded, broken


def zscores(rows: list[dict], mode: str, baseline: set[str]) -> list[dict]:
    """배포 코드의 normalize_by_speaker를 그대로 쓴다 (ser_acted_speech.zscores와 같은 배치 방법)."""
    feats = [r["features"] for r in rows]
    if mode == "none":
        return acoustic.normalize_by_speaker(feats, ["all"] * len(rows), [0.0] * len(rows), baseline_sec=10_000)
    groups = [r["group"] for r in rows]
    if mode == "all":
        return acoustic.normalize_by_speaker(feats, groups, [k * 0.001 for k in range(len(rows))], baseline_sec=10_000)
    if mode == "baseline_neutral_ungated":
        # 진단: 1초 미만이라 '신뢰 발화'가 아니어도 중립 음성을 기준선으로 쓴다. 배포 코드는 기준선에서
        # 신뢰 발화만 쓰므로 hi_kia(대부분 1초 미만)에서는 중립이 아닌 기준선으로 바뀐다. 그 영향을 따로 보려는 것
        feats = [acoustic.SegmentFeatures(values=f.values, duration=max(f.duration, 1.0)) for f in feats]
    starts = [k * 0.001 if r["label"] in baseline else 1000.0 + k for k, r in enumerate(rows)]
    return acoustic.normalize_by_speaker(feats, groups, starts, baseline_sec=30.0)


def safe_auc(pos, neg) -> float | None:
    return auc(pos, neg) if len(pos) and len(neg) else None


def contrast(pairs, pos: set[str], neg: set[str]) -> float | None:
    return safe_auc([a for r, a in pairs if r["label"] in pos], [a for r, a in pairs if r["label"] in neg])


def evaluate(rows: list[dict], arousal: np.ndarray, ds: dict) -> dict:
    pairs = list(zip(rows, arousal))
    out = {"auc_high_vs_low": contrast(pairs, ds["high"], ds["low"])}
    if ds is DATASETS["deeply"]:
        out["auc_negative_vs_neutral"] = contrast(pairs, {"negative"}, {"neutral"})
        out["auc_positive_vs_neutral"] = contrast(pairs, {"positive"}, {"neutral"})
        out["auc_positive_vs_negative"] = contrast(pairs, {"positive"}, {"negative"})
        out["by_room"] = {room: contrast([(r, a) for r, a in pairs if r["room"] == room], ds["high"], ds["low"])
                          for room in sorted({r["room"] for r in rows})}
        out["by_speaker"] = {s: contrast([(r, a) for r, a in pairs if r["speaker"] == s], ds["high"], ds["low"])
                             for s in sorted({r["speaker"] for r in rows})}
    else:
        out["auc_angry_vs_neutral"] = contrast(pairs, {"angry"}, {"neutral"})
        out["by_gender"] = {name: contrast([(r, a) for r, a in pairs if r["female"] == female], ds["high"], ds["low"])
                            for name, female in (("female", True), ("male", False))}
    per_group = [contrast([(r, a) for r, a in pairs if r["group"] == g], ds["high"], ds["low"]) for g in sorted({r["group"] for r in rows})]
    per_group = [a for a in per_group if a is not None]
    out["per_speaker_auc"] = {"min": min(per_group), "median": round(float(np.median(per_group)), 4), "max": max(per_group),
                              "below_0.7": int(sum(a < 0.7 for a in per_group)), "groups": len(per_group)}
    out["mean_arousal_by_label"] = {l: round(float(np.mean([a for r, a in pairs if r["label"] == l])), 3) for l in ds["labels"]}
    return out


def rate_summary(rows: list[dict], ds: dict) -> dict:
    """말 속도 값의 범위. voiced = 규칙이 쓰는 값(음절 / 유성음 프레임 시간), plain = 음절 / 파일 길이 (앞뒤 무음 포함)."""
    def q(values):
        v = np.asarray([x for x in values if not np.isnan(x)])
        return {"p10": round(float(np.percentile(v, 10)), 2), "p50": round(float(np.percentile(v, 50)), 2),
                "p90": round(float(np.percentile(v, 90)), 2), "n": int(len(v))} if len(v) else None
    voiced = [r["features"].values["speech_rate"] for r in rows]
    return {
        "voiced_syll_per_sec": q(voiced),
        "plain_syll_per_sec": q([r["plain_rate"] for r in rows]),
        "nan_share": round(float(np.mean([np.isnan(v) for v in voiced])), 3),
        "voiced_p50_by_label": {l: (q([r["features"].values["speech_rate"] for r in rows if r["label"] == l]) or {}).get("p50")
                                for l in ds["labels"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    rows, broken = load(ds["list"]())
    if not rows:
        raise SystemExit(f"{args.dataset}: 음성을 찾지 못했습니다. 위 docstring의 경로에 압축을 풀어 주세요")
    ser = AcousticArousalBaseline()
    reliable = [r["features"].reliable for r in rows]

    result = {
        "data": {"clips": len(rows), "broken_files": broken, "groups": len({r["group"] for r in rows}),
                 "female_groups": len({r["group"] for r in rows if r["female"]}),
                 "by_label": {l: sum(r["label"] == l for r in rows) for l in ds["labels"]},
                 "reliable_share": round(float(np.mean(reliable)), 3),
                 "duration_sec_p50": round(float(np.median([r["features"].duration for r in rows])), 2)},
        "model": {"name": ser.name, "weights": AROUSAL_WEIGHTS, "note": "가중치를 바꾸지 않고 평가만 함. 한국어 문장을 넘겨 말 속도를 계산"},
        "speech_rate": rate_summary(rows, ds),
        "normalization": {},
        "feature_auc_high_vs_low": {},
        "without_speech_rate": {},
    }
    modes = ("baseline_neutral", "all", "none") + (("baseline_neutral_ungated",) if args.dataset == "hi_kia" else ())
    for mode in modes:
        zs = zscores(rows, mode, ds["baseline"])
        preds = ser.predict(zs, reliable, snr_db=None)
        result["normalization"][mode] = evaluate(rows, np.array([p.arousal for p in preds], dtype=float), ds)
        # 진단: 말 속도 z를 0으로 둔 규칙 (8절 영어 조건과 같음). 말 속도가 보탬이 되는지 본다
        zs0 = [{**z, "speech_rate": 0.0} for z in zs]
        preds0 = ser.predict(zs0, reliable, snr_db=None)
        result["without_speech_rate"][mode] = contrast(list(zip(rows, [p.arousal for p in preds0])), ds["high"], ds["low"])
        if mode == "baseline_neutral":
            for key in acoustic.FEATURES:
                result["feature_auc_high_vs_low"][key] = contrast(list(zip(rows, [z[key] for z in zs])), ds["high"], ds["low"])
            if args.dataset == "deeply":
                result["feature_auc_positive_vs_negative"] = {
                    key: contrast(list(zip(rows, [z[key] for z in zs])), {"positive"}, {"negative"}) for key in acoustic.FEATURES}
            # 신뢰할 수 있는 발화(1초 이상, 유성음 10% 이상)만 남겼을 때
            keep = [i for i, ok in enumerate(reliable) if ok]
            result["reliable_only_high_vs_low"] = contrast([(rows[i], preds[i].arousal) for i in keep], ds["high"], ds["low"])
            if args.dataset == "hi_kia":
                # 청취자 8명 중 5명 이상이 배우의 의도 감정을 맞힌 파일만 (감정이 소리로 전달된 경우)
                agreed = hi_kia_human_agreement()
                keep = [i for i, r in enumerate(rows) if agreed.get(r["fname"])]
                result["human_agreed"] = {"clips": len(keep), "share": round(len(keep) / len(rows), 3),
                                          "share_by_label": {l: round(float(np.mean([bool(agreed.get(r["fname"])) for r in rows if r["label"] == l])), 3)
                                                             for l in ds["labels"]},
                                          "auc_high_vs_low": contrast([(rows[i], preds[i].arousal) for i in keep], ds["high"], ds["low"])}
    save_result(result, RESULTS / f"ser_{args.dataset}.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
