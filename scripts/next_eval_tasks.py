"""다음 평가 세트 절 ID(labeling/eval/naver_shopping_eval_next_ids.json)에서 라벨링 작업 JSON을 다시 만든다.

ID 파일에는 원문이 없으므로, data/external의 네이버 쇼핑 리뷰를 읽어 같은 절을 찾는다.
ID 파일은 별점 순서로 뽑혀 있어서 순서만 봐도 별점이 드러난다. 리뷰 단위로 섞어서 내보낸다(같은 리뷰의 절은 붙여 둔다).

    python scripts/next_eval_tasks.py
    python scripts/make_labeling_page.py data/labeling/next_eval_tasks.json --title "VoC Lens 새 평가 세트 338절" --out data/labeling/next_eval_338.html

원문이 들어 있으므로 data/ 아래에 만들고 저장소에 올리지 않는다.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from voclens.evaluation import read_shopping, to_eval_tasks
from voclens.ingest.review_csv import ReviewRecord
from voclens.pipelines.review import build_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", default="labeling/eval/naver_shopping_eval_next_ids.json")
    ap.add_argument("--shopping", default="data/external/naver_shopping/naver_shopping.txt")
    ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--out", default="data/labeling/next_eval_tasks.json")
    args = ap.parse_args()

    ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))["unit_ids"]
    lines = {int(u.split("#")[0].removeprefix("review:ns")) for u in ids}
    df = read_shopping(args.shopping, split="test")
    df = df[df["line"].isin(lines)]
    units = {}
    for r in df.itertuples():
        session = build_session(ReviewRecord(source_id=f"ns{r.line}", text=r.text, rating=float(r.rating)))
        for u in session.units:
            units[u.unit_id] = {"unit_id": u.unit_id, "text": u.text, "review": session.text}
    missing = [u for u in ids if u not in units]
    if missing:
        raise SystemExit(f"찾지 못한 절 {len(missing)}개 (예: {missing[:3]}). 분할 규칙이나 원문이 바뀌었는지 확인하세요.")

    by_review: dict[str, list[str]] = {}
    for u in ids:
        by_review.setdefault(u.split("#")[0], []).append(u)
    reviews = list(by_review)
    random.Random(args.seed).shuffle(reviews)
    ordered = [units[u] for r in reviews for u in by_review[r]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_eval_tasks(pd.DataFrame(ordered)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"리뷰 {len(reviews)}개, 절 {len(ordered)}개 → {out}")


if __name__ == "__main__":
    main()
