"""리뷰 CSV 읽기.

필수 열: 리뷰 본문 / 선택 열: 별점, 작성일, 브랜드, 매장, 사장님 답글.
열 이름은 configs/default.yaml의 review.columns에서 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReviewRecord:
    source_id: str
    text: str
    rating: float | None = None
    created_at: str | None = None
    brand: str | None = None
    store: str | None = None
    owner_reply: str | None = None


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: UTF-8, CP949로 읽을 수 없습니다")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def load_reviews(path: str | Path, columns: dict[str, str]) -> list[ReviewRecord]:
    path = Path(path)
    df = _read_csv(path)
    text_col = columns["text"]
    if text_col not in df.columns:
        raise ValueError(f"리뷰 본문 열 '{text_col}'이 없습니다. 있는 열: {list(df.columns)}")

    def col(row, key):
        name = columns.get(key)
        return _clean(row[name]) if name and name in df.columns else None

    records = []
    for i, row in df.iterrows():
        text = _clean(row[text_col])
        if not text:
            continue
        rating = col(row, "rating")
        records.append(
            ReviewRecord(
                source_id=f"{path.stem}:{i}",
                text=text,
                rating=float(rating) if rating else None,
                created_at=col(row, "created_at"),
                brand=col(row, "brand"),
                store=col(row, "store"),
                owner_reply=col(row, "owner_reply"),
            )
        )
    return records
