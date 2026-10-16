"""불만 유형 분류기를 약지도로 학습한다 (네이버 쇼핑 개발용 80%만 사용).

약지도 라벨
  양성  규칙의 불만 표현(complaint 정규식, 강도 1.0)이 잡은 절 → 그 유형들
  음성  5점 리뷰에서 불만 표현도 주제 표현도 없고 감성이 칭찬 수준(V ≥ 0.25)인 절 → 불만 없음
개발용 절 정답(labeling/dev/)의 리뷰는 학습에서 빼고, 합치는 방식·임계치 선택에만 쓴다.
평가용 20%와 평가 세트는 전혀 쓰지 않는다.

    python scripts/train_complaint.py                       # CPU 약 1시간
    python scripts/train_complaint.py --reviews 20000 --epochs 1   # 빠른 확인
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_emotion import batches  # noqa: E402  같은 배치 함수를 쓴다

from voclens.config import REPO_ROOT, load_settings  # noqa: E402
from voclens.evaluation import analyze_shopping, read_gold, read_shopping  # noqa: E402
from voclens.text.complaint_model import model_input  # noqa: E402
from voclens.text.taxonomy import KeywordTaxonomyClassifier  # noqa: E402

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
DEV_GOLD = REPO_ROOT / "labeling" / "dev" / "naver_shopping_dev_labels.json"
NONE = "불만 없음"


def weak_dataset(reviews: int, neg_ratio: float, seed: int = 0):
    dev = read_shopping(SRC, "dev")
    held = {int(u.split("#")[0].removeprefix("review:ns")) for u in read_gold(DEV_GOLD)["unit_id"]}
    dev = dev[~dev["line"].isin(held)].sample(reviews, random_state=seed)
    rules = KeywordTaxonomyClassifier()
    labels = sorted(k for k, v in rules.taxonomy.subcategories.items() if not v.get("domains") or "product" in v["domains"])
    pos, neg = [], []
    for s in analyze_shopping(dev, load_settings()):
        prev = ""
        for u in s.units:
            text = u.clean_text or u.text
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
    data = pos + neg
    random.Random(seed + 1).shuffle(data)
    classes = [NONE] + labels
    y = np.array([classes.index(ls[0]) if ls else 0 for _, ls in data], dtype=np.int64)
    return [t for t, _ in data], y, classes, len(pos), len(neg)


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
    args = ap.parse_args()
    torch.manual_seed(0)

    texts, y, labels, n_pos, n_neg = weak_dataset(args.reviews, args.neg_ratio)
    (Path(args.out)).mkdir(parents=True, exist_ok=True)
    counts = dict(zip(labels, np.bincount(y, minlength=len(labels)).tolist()))
    print(f"약지도 절 {len(texts):,}개 (불만 {n_pos:,} · 불만 없음 {n_neg:,}) · 유형별 {counts}", flush=True)

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
    (out / "training.json").write_text(json.dumps({"base": args.base, "args": vars(args), "positives": n_pos, "negatives": n_neg,
                                                    "label_counts": counts, "seconds": round(time.time() - start)},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
