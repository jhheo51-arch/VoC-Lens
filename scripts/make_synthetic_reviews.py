"""데모용 합성 배달 리뷰 생성기.

실제 리뷰가 아니다. 파이프라인과 리포트를 시연하려고 만든 가상 데이터이며,
브랜드 이름도 가상이다. 시드를 고정해서 같은 파일이 다시 만들어진다.

  python scripts/make_synthetic_reviews.py

설계
- 치킨브랜드A: 배달 지연·식은 음식이 주요 불만
- 피자브랜드B: 누락·응대 불만, 10월에 누락 불만이 늘어나도록 설정 (추세 탐지 시연)
- 리뷰 이벤트 리뷰는 불만이 있어도 별점 5점을 주는 경우를 섞음 (별점–감정 불일치 시연)
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20261014
OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "delivery_reviews_synthetic.csv"

POSITIVE = [
    "치킨이 정말 바삭하고 맛있어요",
    "양도 푸짐해요",
    "배달이 생각보다 빨리 왔어요",
    "사장님이 친절하세요",
    "포장이 깔끔했어요",
    "가격 대비 만족스러워요",
    "소스가 진짜 맛있어요",
    "따뜻하게 도착했어요",
    "도우가 쫄깃하고 토핑이 넉넉해요",
    "아이들이 너무 좋아해요ㅎㅎ",
]
POSITIVE_CLOSERS = ["또 시킬게요", "재주문 의사 있어요", "친구들한테 강추합니다", "단골 될 것 같아요", "최고예요 👍", ""]

NEGATIVE = {
    "지연": ["한 시간 넘게 걸려서 왔어요", "예상 시간보다 한참 늦게 도착했어요", "배달이 너무 늦었어요", "1시간 만에 왔네요 기다리다 지쳤어요"],
    "누락": ["콜라가 누락됐어요", "치즈볼이 빠져 있었어요", "주문한 거랑 다른 메뉴가 왔어요", "사이드가 안 들어 있었어요"],
    "온도": ["다 식어서 왔어요", "튀김이 눅눅해요", "피자가 차갑게 와서 아쉬웠어요"],
    "맛": ["너무 짜서 못 먹었어요", "소스가 너무 짜요", "맛이 예전 같지 않아요"],
    "양": ["가격에 비해 양이 너무 적어요", "양이 예전보다 줄었어요"],
    "포장": ["소스가 다 새서 왔어요", "포장이 뜯어져 있었어요"],
    "연락": ["가게에 전화했는데 전화를 안 받아요", "몇 번이나 연락했는데 답변이 없어요"],
    "응대": ["사장님이 너무 불친절해요", "문의했더니 짜증을 내셨어요"],
    "위생": ["음식에서 머리카락이 나왔어요 찝찝해요"],
    "가격": ["이 가격이면 너무 비싸요", "배달비가 너무 비싸요"],
    "요청": ["피클 빼달라고 했는데 들어 있어요", "요청사항 적었는데 무시하셨네요"],
}
NEGATIVE_CLOSERS = ["다신 안 시켜요", "실망이에요ㅠㅠ", "비추합니다", "재주문은 안 할 것 같아요", "화가 나네요", ""]
MIXED_OPENERS = ["맛은 좋은데", "치킨은 맛있는데", "양은 많은데", "가격은 착한데", "포장은 깔끔했는데"]
RECOVERY = [
    "처음엔 콜라가 빠져서 화났는데 사장님이 바로 다시 보내주셨어요 감사합니다",
    "배달이 늦어서 속상했지만 사장님이 사과 전화 주시고 환불해주셨어요",
    "음식이 다 식어서 실망했는데 사장님이 바로 다시 만들어 보내주셨어요 친절하세요",
    "요청사항이 반영 안 돼서 짜증났는데 사장님이 죄송하다고 서비스까지 챙겨주셨어요",
]
EVENT_PREFIX = ["리뷰이벤트 참여합니다! ", "서비스 감사합니다 ", "리뷰 약속하고 치즈볼 서비스 받았어요 "]
OWNER_REPLIES = {
    "positive": "맛있게 드셔서 감사합니다! 또 찾아주세요.",
    "negative": "불편을 드려 죄송합니다. 확인 후 개선하겠습니다.",
}

BRANDS = {
    "치킨브랜드A": {"stores": ["강남점", "홍대점", "잠실점"], "issues": {"지연": 5, "온도": 4, "맛": 2, "양": 2, "가격": 2, "포장": 1, "응대": 1}},
    "피자브랜드B": {"stores": ["역삼점", "신촌점", "분당점"], "issues": {"누락": 5, "연락": 3, "응대": 2, "요청": 2, "지연": 1, "위생": 1, "가격": 1}},
}


def _pick_issue(rng: random.Random, brand: str, day: date) -> str:
    issues = dict(BRANDS[brand]["issues"])
    if brand == "피자브랜드B" and day >= date(2026, 10, 1):
        issues["누락"] *= 3  # 10월에 누락 불만 급증
    names, weights = zip(*issues.items())
    return rng.choices(names, weights=weights)[0]


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def make_review(rng: random.Random, brand: str, day: date) -> dict:
    kind = rng.choices(["positive", "negative", "mixed", "recovery"], weights=[45, 30, 18, 7])[0]
    issue = _pick_issue(rng, brand, day)
    neg = rng.choice(NEGATIVE[issue])

    if kind == "positive":
        text = _join(*rng.sample(POSITIVE, k=rng.randint(1, 3)), rng.choice(POSITIVE_CLOSERS))
        rating = rng.choice([5, 5, 5, 4])
    elif kind == "negative":
        extra = rng.choice(NEGATIVE[_pick_issue(rng, brand, day)]) if rng.random() < 0.3 else ""
        text = _join(neg, extra, rng.choice(NEGATIVE_CLOSERS))
        rating = rng.choice([1, 1, 2, 2, 3])
    elif kind == "mixed":
        text = _join(rng.choice(MIXED_OPENERS), neg, rng.choice(["", "아쉬워요", "다음엔 신경 써주세요"]))
        rating = rng.choice([3, 3, 4, 4, 2])
    else:
        text = rng.choice(RECOVERY)
        rating = rng.choice([4, 5, 5])

    review_event = rng.random() < 0.15
    if review_event:
        text = rng.choice(EVENT_PREFIX) + text
        rating = 5

    reply = ""
    if rng.random() < 0.4:
        reply = OWNER_REPLIES["positive" if rating >= 4 else "negative"]

    return {
        "text": text,
        "rating": rating,
        "created_at": day.isoformat(),
        "brand": brand,
        "store": rng.choice(BRANDS[brand]["stores"]),
        "owner_reply": reply,
    }


def main(n_per_brand: int = 90) -> None:
    rng = random.Random(SEED)
    start, days = date(2026, 8, 1), 71
    rows = []
    for brand in BRANDS:
        for _ in range(n_per_brand):
            rows.append(make_review(rng, brand, start + timedelta(days=rng.randrange(days))))
    rows.sort(key=lambda r: r["created_at"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}건 → {OUT}")


if __name__ == "__main__":
    main()
