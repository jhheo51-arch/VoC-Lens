"""KOTE로 감정 분류 모델을 파인튜닝한다 (CPU에서도 돌아가는 작은 모델이 기본값).

    python scripts/fetch_public_data.py kote
    python scripts/train_emotion.py                                    # koelectra-small, CPU 약 30분~1시간
    python scripts/train_emotion.py --base beomi/KcELECTRA-base-v2022 --epochs 3 --out models/emotion-kote-base   # Colab GPU 권장

KOTE 라벨(44감정)은 VoC Lens 6감정 묶음으로 바꿔 다중 라벨로 학습한다 (voclens.evaluation.KOTE_TO_EMOTION).
test 5천 문장은 학습에 쓰지 않는다. 검증은 val, 최종 비교는 `voclens eval kote --model`로 test에서 한다.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from voclens.config import REPO_ROOT
from voclens.evaluation import KOTE_TO_EMOTION, read_kote
from voclens.text.emotion_model import GROUPS
from voclens.text.preprocess import preprocess

KOTE = REPO_ROOT / "data" / "external" / "kote"


def targets(labels: list[list[str]]) -> np.ndarray:
    return np.array([[float(bool(set(ls) & set(KOTE_TO_EMOTION[g]))) for g in GROUPS] for ls in labels], dtype=np.float32)


def batches(tokenizer, texts, y, batch_size, max_length, shuffle, seed=0):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        enc = tokenizer([texts[j] for j in b], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        yield enc, torch.from_numpy(y[b])


def evaluate(model, tokenizer, texts, y, batch_size, max_length):
    model.eval()
    probs = []
    with torch.no_grad():
        for enc, _ in batches(tokenizer, texts, y, batch_size, max_length, shuffle=False):
            probs.append(torch.sigmoid(model(**enc).logits).numpy())
    p = np.vstack(probs)
    pred = p >= 0.5
    f1s = []
    for k in range(len(GROUPS)):
        tp = float((pred[:, k] & (y[:, k] == 1)).sum())
        prec = tp / max(1, pred[:, k].sum())
        rec = tp / max(1, (y[:, k] == 1).sum())
        f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    model.train()
    return {g: round(float(f), 3) for g, f in zip(GROUPS, f1s)} | {"macro_f1": round(float(np.mean(f1s)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="monologg/koelectra-small-v3-discriminator")
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "emotion-kote-small"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--limit", type=int, help="학습 문장 수 제한 (빠른 확인용)")
    args = ap.parse_args()
    torch.manual_seed(0)

    train, val = read_kote(KOTE / "train.tsv"), read_kote(KOTE / "val.tsv")
    if args.limit:
        train = train.sample(args.limit, random_state=0)
    tx_train = [preprocess(t).text for t in train["text"]]
    tx_val = [preprocess(t).text for t in val["text"]]
    y_train, y_val = targets(train["labels"].tolist()), targets(val["labels"].tolist())
    print(f"train {len(tx_train):,} · val {len(tx_val):,} · 감정별 양성 비율 {dict(zip(GROUPS, y_train.mean(0).round(2)))}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(GROUPS), problem_type="multi_label_classification",
        id2label=dict(enumerate(GROUPS)), label2id={g: i for i, g in enumerate(GROUPS)},
    )
    steps = args.epochs * ((len(tx_train) + args.batch_size - 1) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    step, start, history = 0, time.time(), []
    for epoch in range(args.epochs):
        for enc, yb in batches(tokenizer, tx_train, y_train, args.batch_size, args.max_length, shuffle=True, seed=epoch):
            loss = model(**enc, labels=yb).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} · loss {loss.item():.4f} · {time.time() - start:.0f}s", flush=True)
        metrics = evaluate(model, tokenizer, tx_val, y_val, 128, args.max_length)
        history.append({"epoch": epoch + 1, **metrics})
        print(f"epoch {epoch + 1} val {metrics}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training.json").write_text(json.dumps({"base": args.base, "args": vars(args), "train_size": len(tx_train),
                                                    "seconds": round(time.time() - start), "val": history},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {out}")


if __name__ == "__main__":
    main()
