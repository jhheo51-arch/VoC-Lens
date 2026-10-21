"""불만 유형 분류기를 약지도로 학습한다 (네이버 쇼핑 개발용 80%만 사용).

약지도 라벨
  양성  규칙의 불만 표현(complaint 정규식, 강도 1.0)이 잡은 절 → 그 유형들
  음성  5점 리뷰에서 불만 표현도 주제 표현도 없고 감성이 칭찬 수준(V ≥ 0.25)인 절 → 불만 없음
  직접  --direct-labels로 준 개발용 직접 라벨(절 ID와 유형). 같은 절의 약지도 라벨을 대신하고 --direct-repeat번 넣는다
개발용 절 정답(labeling/dev/naver_shopping_dev_labels.json)의 리뷰는 학습에서 빼고, 합치는 방식·임계치 선택에만 쓴다.
평가용 20%와 평가 세트는 전혀 쓰지 않는다.

    python scripts/train_complaint.py                       # CPU 약 1시간
    python scripts/train_complaint.py --reviews 20000 --epochs 1   # 빠른 확인
    python scripts/train_complaint.py --epochs 3 --direct-labels labeling/dev/naver_shopping_dev_direct_labels.json \\
        --out models/complaint-weak-direct-small
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_emotion import batches  # noqa: E402  같은 배치 함수를 쓴다

from voclens.config import REPO_ROOT, load_settings  # noqa: E402
from voclens.evaluation import analyze_shopping, read_gold, read_shopping, split_of  # noqa: E402
from voclens.text.complaint_model import model_input  # noqa: E402
from voclens.text.taxonomy import KeywordTaxonomyClassifier  # noqa: E402

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
DEV_GOLD = REPO_ROOT / "labeling" / "dev" / "naver_shopping_dev_labels.json"
NONE = "불만 없음"
NON_COMPLAINT = ("칭찬", "해당 없음", "분류 불가")


def line_of(unit_id: str) -> int:
    return int(unit_id.split("#")[0].removeprefix("review:ns"))


def direct_examples(gold, inputs: dict[str, str], classes: list[str]) -> tuple[list[tuple[str, list[str]]], Counter]:
    """직접 라벨(read_gold 형식) → [(모델 입력, [유형])]. 보류는 빼고, 불만 아님(칭찬·해당 없음·분류 불가)은
    "불만 없음"으로, 불만 유형이 여러 개면 유형마다 한 번씩 넣는다. 분류기에 없는 유형은 건너뛰고 센다."""
    out, skipped = [], Counter()
    for r in gold[~gold["hold"]].itertuples():
        if r.unit_id not in inputs:
            skipped["절을 찾지 못함"] += 1
            continue
        types = [l for l in r.labels if l not in NON_COMPLAINT]
        if not types:
            out.append((inputs[r.unit_id], []))
        for l in types:
            if l in classes:
                out.append((inputs[r.unit_id], [l]))
            else:
                skipped[l] += 1
    return out, skipped


def direct_dataset(paths: list[str], classes: list[str]):
    """직접 라벨 파일들을 읽어 절을 다시 나누고 모델 입력(앞 절 문맥 포함)을 만든다. 개발용 분할인지 확인한다."""
    gold = pd.concat([read_gold(p) for p in paths], ignore_index=True).drop_duplicates("unit_id", keep="last")
    lines = {line_of(u) for u in gold["unit_id"]}
    held = {line_of(u) for u in read_gold(DEV_GOLD)["unit_id"]}
    assert all(split_of(l) == "dev" for l in lines), "직접 라벨에 평가용 분할 리뷰가 있다"
    assert not lines & held, "직접 라벨에 개발용 절 정답(임계치 선택용)의 리뷰가 있다"
    shop = read_shopping(SRC, "dev")
    inputs = {}
    for s in analyze_shopping(shop[shop["line"].isin(lines)], load_settings()):
        prev = ""
        for u in s.units:
            text = u.clean_text or u.text
            inputs[u.unit_id] = model_input(text, prev)
            prev = text
    examples, skipped = direct_examples(gold, inputs, classes)
    return examples, set(gold["unit_id"]), skipped


def weak_dataset(reviews: int, neg_ratio: float, seed: int = 0, direct_paths: list[str] | None = None, direct_repeat: int = 1):
    dev = read_shopping(SRC, "dev")
    held = {line_of(u) for u in read_gold(DEV_GOLD)["unit_id"]}
    dev = dev[~dev["line"].isin(held)].sample(reviews, random_state=seed)
    rules = KeywordTaxonomyClassifier()
    labels = sorted(k for k, v in rules.taxonomy.subcategories.items() if not v.get("domains") or "product" in v["domains"])
    classes = [NONE] + labels
    direct, direct_ids, direct_info = [], set(), {}
    if direct_paths:
        direct, direct_ids, skipped = direct_dataset(direct_paths, classes)
        direct_info = {"units": len(direct_ids), "examples": len(direct), "repeat": direct_repeat,
                       "label_counts": dict(Counter(ls[0] if ls else NONE for _, ls in direct)), "skipped": dict(skipped)}
    pos, neg = [], []
    for s in analyze_shopping(dev, load_settings()):
        prev = ""
        for u in s.units:
            text = u.clean_text or u.text
            if u.unit_id in direct_ids:          # 직접 라벨이 있는 절은 약지도 라벨을 쓰지 않는다
                prev = text
                continue
            hits = rules.classify(text, 0.0, "product", prev)          # V=0 → 주제 표현은 빼고 불만 표현만
            strong = [l for l, w in hits if w >= 1.0]
            if strong:
                pos.append((model_input(text, prev), strong))
            # V=-1이면 주제 표현까지 검사한다. 불만 표현·주제 표현이 모두 없는 5점 리뷰의 칭찬 절만 음성으로 쓴다
            elif s.meta["rating"] == 5 and (u.valence or 0) >= 0.25 and not rules.classify(text, -1.0, "product", prev):
                neg.append((model_input(text, prev), []))
            prev = text
    random.Random(seed).shuffle(neg)
    # 22개 중 하나를 고르는 방식(softmax)으로 학습한다. 유형마다 따로 맞다/아니다를 배우면(시그모이드)
    # 양성이 유형마다 10% 미만이라 "전부 아니다"로 쏠렸다. 여러 유형이 붙은 절(약 4%)은 유형마다 한 번씩 넣는다
    pos = [(t, [l]) for t, ls in pos for l in ls]
    neg = neg[: int(len(pos) * neg_ratio)]
    data = pos + neg + direct * direct_repeat
    random.Random(seed + 1).shuffle(data)
    y = np.array([classes.index(ls[0]) if ls else 0 for _, ls in data], dtype=np.int64)
    return [t for t, _ in data], y, classes, len(pos), len(neg), direct_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="monologg/koelectra-small-v3-discriminator")
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "complaint-weak-small"))
    ap.add_argument("--reviews", type=int, default=60000, help="약지도 라벨을 만들 개발용 리뷰 수")
    ap.add_argument("--neg-ratio", type=float, default=1.5, help="양성 대비 음성 비율")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--direct-labels", nargs="*", default=[], help="개발용 직접 라벨 JSON([{unit_id, labels, hold}])")
    ap.add_argument("--direct-repeat", type=int, default=5, help="직접 라벨 절을 몇 번 넣을지 (약지도 2만여 절에 묻히지 않게)")
    args = ap.parse_args()
    torch.manual_seed(0)

    texts, y, labels, n_pos, n_neg, direct_info = weak_dataset(args.reviews, args.neg_ratio, direct_paths=args.direct_labels,
                                                               direct_repeat=args.direct_repeat)
    (Path(args.out)).mkdir(parents=True, exist_ok=True)
    counts = dict(zip(labels, np.bincount(y, minlength=len(labels)).tolist()))
    print(f"학습 절 {len(texts):,}개 (약지도 불만 {n_pos:,} · 불만 없음 {n_neg:,} · 직접 {direct_info}) · 유형별 {counts}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(labels), problem_type="single_label_classification",
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
    )
    steps = args.epochs * ((len(texts) + args.batch_size - 1) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    step, start = 0, time.time()
    for epoch in range(args.epochs):
        for enc, yb in batches(tokenizer, texts, y, args.batch_size, args.max_length, shuffle=True, seed=epoch):
            loss = model(**enc, labels=yb).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} · loss {loss.item():.4f} · {time.time() - start:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")   # 0번은 "불만 없음"
    (out / "training.json").write_text(json.dumps({"base": args.base, "args": vars(args), "positives": n_pos, "negatives": n_neg, "direct": direct_info,
                                                    "label_counts": counts, "seconds": round(time.time() - start)},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
