"""YouTube 영상을 한 번 받아 번들 JSON으로 data/external/youtube/ 에 둔다 (같은 영상을 여러 설정으로 다시 돌릴 때 재수집하지 않으려고).

자막·댓글 원문이 들어 있으므로 data/ 아래(.gitignore)에만 두고 저장소에 올리지 않는다.
음성은 남기지 않는다 (--stt로 전사한 영상도 전사 결과만 저장).

    python scripts/video_bundle_cache.py https://www.youtube.com/watch?v=dStfwAbQB0A https://www.youtube.com/watch?v=-D9zBKMBkeU
    python scripts/video_bundle_cache.py --stt https://www.youtube.com/watch?v=ULoEZw889Lc
"""

import argparse
import time

from voclens.config import REPO_ROOT
from voclens.ingest.youtube import fetch_bundle, video_id_from_url

CACHE = REPO_ROOT / "data" / "external" / "youtube"


def cache_path(url_or_id: str):
    vid = url_or_id if len(url_or_id) == 11 and "/" not in url_or_id else video_id_from_url(url_or_id)
    return CACHE / f"{vid}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--stt", action="store_true", help="사람 자막이 없으면 음성 전사·화자 분리")
    ap.add_argument("--force", action="store_true", help="이미 받은 영상도 다시 받기")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for url in args.urls:
        out = cache_path(url)
        if out.exists() and not args.force:
            print(f"{out.name}: 이미 있음")
            continue
        t0 = time.perf_counter()
        bundle = fetch_bundle(url, allow_stt=args.stt)
        bundle.audio = None
        out.write_text(bundle.model_dump_json(exclude={"source_path"}), encoding="utf-8")
        print(f"{out.name}: {bundle.transcript_source}, 문장 {len(bundle.segments)}개, 댓글 {len(bundle.comments)}개 ({time.perf_counter() - t0:.0f}초)")


if __name__ == "__main__":
    main()
