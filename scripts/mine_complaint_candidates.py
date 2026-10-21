"""재현율이 0인 불만 유형(성능, 오배송·누락, 효과 없음)의 직접 라벨 후보 절을 개발용 80%에서 고른다.

규칙(정규식)만으로 후보를 고르면 규칙이 못 잡는 표현이 다시 빠지므로, 아래 네 가지를 섞는다.
  kw_low    1~2점 리뷰에서 넓은 검색어(이 스크립트에 따로 적은 것, 분류 규칙 정규식이 아님)가 걸리고 규칙은 그 유형을 못 잡은 절
  kw_high   4~5점 리뷰에서 같은 검색어가 걸린 절 (비슷하지만 불만이 아닌 음성 후보)
  emb       유형 정의의 예문과 바꿔 쓴 예문(SEEDS)에 임베딩이 가까운 절 (koelectra-small 평균 풀링, 규칙을 못 잡은 절만)
  model     기존 약지도 분류기가 그 유형에 0.1~0.9 확률을 준 절 (기준값 아래라 놓친 것)
평가용 20%와 개발용 절 정답(labeling/dev/naver_shopping_dev_labels.json)의 리뷰는 쓰지 않는다.

출력은 원문이 들어 있어 data/labeling/ 아래에 둔다(저장소에 올리지 않음).
  data/labeling/direct_candidates.json   후보 + 출처(source, type_hint, 별점, 규칙 예측) — 라벨러에게 주지 않는다
  data/labeling/direct_tasks.json        라벨러용: unit_id, 절, 리뷰 전체만 (순서 섞음, 출처·별점·규칙 예측 없음)

    python scripts/mine_complaint_candidates.py                       # 1차: 네 가지 출처
    python scripts/mine_complaint_candidates.py --round 2             # 2차: 좁힌 검색어(SEARCH_R2)로 1~2점·4점 리뷰만

1차 라벨 결과, 임베딩(emb)과 기존 분류기(model) 후보에서는 세 유형이 거의 나오지 않았다(emb 120개 중 1개).
그래서 2차는 검색어를 좁혀 1~2점(kw_low)과 4점(kw_mid, "좋은데 소음이 좀" 같은 섞인 절) 리뷰에서 규칙이 못 잡은 절만 더 모은다.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from voclens.config import REPO_ROOT, load_settings
from voclens.evaluation import analyze_shopping, read_gold, read_shopping
from voclens.text.complaint_model import model_input

SRC = REPO_ROOT / "data" / "external" / "naver_shopping" / "naver_shopping.txt"
DEV_GOLD = REPO_ROOT / "labeling" / "dev" / "naver_shopping_dev_labels.json"
TYPES = ["성능", "오배송·누락", "효과 없음"]

# 후보를 넓게 모으는 검색어. 분류 규칙 정규식이 아니며, 일부러 넓게 잡아 불만이 아닌 절도 걸리게 한다
SEARCH = {
    "성능": r"소음|시끄|조용|흡입|빨아들|안 ?빨|느려|느리|속도|버벅|끊기|끊겨|렉|정확|오차|엉터리|화력|안 ?데워|미지근|"
            r"바람이 ?약|약해요|힘이 ?없|출력|성능|안 ?나와|잘 ?안 ?돼|안 ?잘려|잘 ?안 ?잘|안 ?갈려|인식|연결이|"
            r"반응|감도|온도|뜨거워지|안 ?시원|안 ?따뜻|열이 ?안|약하",
    "오배송·누락": r"다른 ?(상품|제품|게|걸|색|사이즈|옵션)|잘못 ?(왔|와|보내|배송)|빠져 ?있|빠졌|누락|안 ?들어|없네|없어요|"
                 r"하나만|한 ?개만|덜 ?왔|모자라|부족|구성품|사은품|증정|개수|갯수|수량|옵션.{0,6}(다르|틀리|잘못)|"
                 r"주문한 ?(거|것|게)|시킨 ?(거|것|게)|안 ?왔|안 ?와요|두 ?개|세트",
    "효과 없음": r"효과|효능|모르겠|그대로|변화|차이(가|를)? ?(없|모르|못)|똑같|소용|안 ?지워|안 ?빠져|안 ?빠지|보습|촉촉|"
               r"건조|각질|트러블|탈모|미백|흡수|세정|때가|얼룩|살 ?안 ?빠|체중|잠이|개선|좋아지",
}

# 2차 검색어: 1차보다 좁혀서 실제 불만 비율을 높인다 (역시 분류 규칙 정규식이 아님)
SEARCH_R2 = {
    "성능": r"소음|시끄|흡입|빨아들|안 ?빨|느려|느리|버벅|끊겨|끊기|오차|엉터리|화력|안 ?데워|미지근|바람이 ?약|힘이 ?없|"
            r"출력|성능|안 ?잘려|안 ?갈려|인식이? ?안|연결이? ?안|반응이? ?(?:느|없|안)|감도|안 ?시원|안 ?따뜻|열이 ?안|"
            r"약해서|약해요|안 ?나와|흡착|뜨거워지지|온도가 ?안",
    "오배송·누락": r"다른 ?(?:상품|제품|게|걸|색상?|사이즈|옵션|모델)|잘못 ?(?:왔|와|보내|배송|발송)|빠져 ?(?:있|왔)|빠졌|누락|"
                 r"안 ?들어 ?(?:있|왔)|들어있지 ?않|하나만 ?(?:왔|와|들어)|한 ?개만|덜 ?(?:왔|와|들어)|모자라|구성품|사은품|"
                 r"증정품|개수|갯수|수량|(?:주문한|시킨) ?(?:거|것|게|제품|상품)(?:랑|과|이|하고)|안 ?보내|설명서가? ?없|"
                 r"[12]개만|짝짝이|한 ?짝",
    "효과 없음": r"효과|효능|모르겠|변화가? ?없|차이(?:가|를)? ?(?:없|모르|못)|소용|안 ?지워|안 ?빠져|안 ?빠지|"
               r"보습이? ?(?:안|부족|전혀)|각질|트러블|탈모|흡수가? ?안|세정력|살이? ?안 ?빠|체중|개선이? ?안|"
               r"그대로(?:예요|네요|임|에요|입니다)",
}
R2_QUOTA = {"성능": {"kw_low": 150, "kw_mid": 40}, "오배송·누락": {"kw_low": 150, "kw_mid": 40},
            "효과 없음": {"kw_low": 80, "kw_mid": 20}}

# 임베딩 검색용 예문: label_definitions_product.md의 예와 같은 뜻을 바꿔 쓴 것
SEEDS = {
    "성능": ["소음이 너무 커요", "온도계가 엉터리예요", "흡입력이 약해서 먼지가 안 빨려요", "속도가 너무 느려요",
            "바람이 약해서 시원하지 않아요", "잘 안 잘려요", "물이 잘 안 데워져요", "인식이 잘 안 돼요",
            "생각보다 성능이 별로예요", "힘이 너무 약해요"],
    "오배송·누락": ["블루를 샀는데 블랙이 왔어요", "두 개 중 하나만 왔어요", "구성품이 빠져 있어요", "사은품이 안 왔어요",
                  "다른 상품이 배송됐어요", "주문한 사이즈가 아닌 다른 사이즈가 왔어요", "수량이 모자라요",
                  "옵션이 잘못 왔어요", "하나가 안 들어 있네요", "설명서가 없어요"],
    "효과 없음": ["바르나 안 바르나 똑같아요", "효과를 모르겠어요", "써도 변화가 없어요", "냄새 제거가 전혀 안 돼요",
                "때가 하나도 안 지워져요", "보습이 전혀 안 돼요", "먹어도 아무 효과가 없어요", "얼룩이 그대로예요",
                "한 달 써도 달라진 게 없어요", "기대했던 효과가 없네요"],
}


def clause_rows(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    rows = []
    for s in analyze_shopping(df, settings):
        prev = ""
        for u in s.units:
            text = u.clean_text or u.text
            rows.append({"unit_id": u.unit_id, "text": u.text, "input": model_input(text, prev), "review": s.text,
                         "rating": int(s.meta["rating"]), "rules": list(u.complaint_categories)})
            prev = text
    return pd.DataFrame(rows)


def embed(texts: list[str], base: str, batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok, model = AutoTokenizer.from_pretrained(base), AutoModel.from_pretrained(base).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1)).numpy()
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--pool-reviews", type=int, default=12000, help="임베딩·모델 검색에 쓸 무작위 개발용 리뷰 수")
    ap.add_argument("--kw-reviews", type=int, default=1500, help="유형·별점대마다 검색어로 걸린 리뷰 중 볼 수")
    ap.add_argument("--quota", default="kw_low=60,kw_high=25,emb=40,model=20")
    ap.add_argument("--complaint-model", default=str(REPO_ROOT / "models" / "complaint-weak-small"))
    ap.add_argument("--embed-base", default="monologg/koelectra-small-v3-discriminator")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "labeling"))
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    quota = {k: int(v) for k, v in (p.split("=") for p in args.quota.split(","))}
    rng = random.Random(args.seed)
    settings = load_settings()

    dev = read_shopping(SRC, "dev")
    held = {int(u.split("#")[0].removeprefix("review:ns")) for u in read_gold(DEV_GOLD)["unit_id"]}
    dev = dev[~dev["line"].isin(held)].reset_index(drop=True)

    picked: dict[str, dict] = {}
    out = Path(args.out_dir)
    if args.round == 2:
        args.seed += 100
        rng = random.Random(args.seed)
        before = {c["unit_id"] for c in json.loads((out / "direct_candidates.json").read_text(encoding="utf-8"))}

    def take(df: pd.DataFrame, n: int, source: str, kind: str) -> None:
        df = df[~df["unit_id"].isin(picked)]
        for r in df.sample(min(n, len(df)), random_state=rng.randrange(10**6)).itertuples():
            picked[r.unit_id] = {"unit_id": r.unit_id, "text": r.text, "review": r.review, "rating": r.rating,
                                 "rules": r.rules, "source": source, "type_hint": kind}

    if args.round == 2:
        for kind in TYPES:
            pat = re.compile(SEARCH_R2[kind])
            hit = dev[dev["text"].str.contains(SEARCH_R2[kind], regex=True)]
            for source, cond in (("kw_low", hit["rating"] <= 2), ("kw_mid", hit["rating"] == 4)):
                sub = hit[cond]
                sub = sub.sample(min(args.kw_reviews * 2, len(sub)), random_state=args.seed)
                c = clause_rows(sub, settings)
                c = c[c["text"].map(lambda t: bool(pat.search(t))) & c["rules"].map(lambda ls: kind not in ls)
                      & ~c["unit_id"].isin(before)]
                take(c, R2_QUOTA[kind][source], source, kind)
            print(kind, "2차 후보", Counter(v["source"] for v in picked.values() if v["type_hint"] == kind), flush=True)
        save(list(picked.values()), rng, out, "_r2")
        return

    # 1) 검색어
    for kind in TYPES:
        pat = re.compile(SEARCH[kind])
        hit = dev[dev["text"].str.contains(SEARCH[kind], regex=True)]
        for source, cond in (("kw_low", hit["rating"] <= 2), ("kw_high", hit["rating"] >= 4)):
            sub = hit[cond]
            sub = sub.sample(min(args.kw_reviews, len(sub)), random_state=args.seed)
            c = clause_rows(sub, settings)
            c = c[c["text"].map(lambda t: bool(pat.search(t)))]
            if source == "kw_low":
                c = c[c["rules"].map(lambda ls: kind not in ls)]
            take(c, quota[source], source, kind)
        print(kind, "검색어 후보", Counter(v["source"] for v in picked.values() if v["type_hint"] == kind), flush=True)

    # 2) 임베딩·기존 분류기 (무작위 개발용 리뷰 풀)
    pool = clause_rows(dev.sample(args.pool_reviews, random_state=args.seed + 1), settings)
    pool = pool[pool["text"].str.len() >= 4].reset_index(drop=True)
    print(f"풀 절 {len(pool):,}개", flush=True)
    vec = embed(pool["input"].str.split(" \\[SEP\\] ").str[-1].tolist(), args.embed_base)
    from voclens.text.complaint_model import TransformerComplaintClassifier

    clf = TransformerComplaintClassifier(args.complaint_model)
    probs = clf.probs(pool["input"].tolist())
    for kind in TYPES:
        missed = pool["rules"].map(lambda ls: kind not in ls).to_numpy()
        seed_vec = embed(SEEDS[kind], args.embed_base)
        sim = (vec @ seed_vec.T).max(1)
        cand = pool[missed].assign(sim=sim[missed]).sort_values("sim", ascending=False).head(quota["emb"] * 4)
        take(cand, quota["emb"], "emb", kind)
        p = probs[:, clf.labels.index(kind)]
        band = pool[missed & (p >= 0.1) & (p < 0.9)]
        take(band, quota["model"], "model", kind)

    save(list(picked.values()), rng, out, "")


def save(cands: list[dict], rng: random.Random, out: Path, suffix: str) -> None:
    rng.shuffle(cands)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"direct_candidates{suffix}.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    tasks = [{"unit_id": c["unit_id"], "text": c["text"], "review": c["review"]} for c in cands]
    (out / f"direct_tasks{suffix}.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"후보 {len(cands)}개", Counter((c["type_hint"], c["source"]) for c in cands))


if __name__ == "__main__":
    main()
