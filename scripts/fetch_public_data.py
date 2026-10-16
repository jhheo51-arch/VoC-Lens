"""공개 한국어 데이터셋을 data/external/ 로 내려받는다.

원문은 재배포하지 않는다 (data/ 는 .gitignore). 저장소에는 이 스크립트와 출처·라이선스 표기만 둔다.

    python scripts/fetch_public_data.py            # 전부
    python scripts/fetch_public_data.py kote       # 하나만
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"

DATASETS = {
    "naver_shopping": {
        "license": "Public Domain (bab2min/corpus README)",
        "source": "https://github.com/bab2min/corpus",
        "files": {
            "naver_shopping.txt": "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
        },
    },
    "kote": {
        "license": "MIT (Copyright (c) 2022 Jeon Duyoung)",
        "source": "https://github.com/searle-j/KOTE",
        "files": {
            name: f"https://raw.githubusercontent.com/searle-j/KOTE/main/{name}"
            for name in ("train.tsv", "val.tsv", "test.tsv", "LICENSE")
        },
    },
}


def fetch(key: str) -> None:
    spec = DATASETS[key]
    target = OUT / key
    target.mkdir(parents=True, exist_ok=True)
    for name, url in spec["files"].items():
        path = target / name
        if path.exists():
            print(f"있음  {path.relative_to(ROOT)}")
            continue
        urllib.request.urlretrieve(url, path)
        print(f"받음  {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f}MB)")
    (target / "SOURCE.txt").write_text(f"{spec['source']}\n{spec['license']}\n", encoding="utf-8")


if __name__ == "__main__":
    for key in sys.argv[1:] or DATASETS:
        fetch(key)
