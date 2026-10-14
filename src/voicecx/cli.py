"""voicecx 명령줄 도구.

  voicecx review data/samples/delivery_reviews_synthetic.csv
  voicecx report --out reports/review_report.md
  voicecx query --list
  voicecx query complaint_share_by_brand
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voicecx.config import load_settings
from voicecx.schema import INDEX_LABELS


def _print_table(rows, max_width: int = 40) -> None:
    if not rows:
        print("(결과 없음)")
        return
    headers = list(rows[0].keys())
    cells = [[("" if r[h] is None else str(r[h]))[:max_width] for h in headers] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths)))


def cmd_review(args, settings) -> int:
    from voicecx.pipelines.review import run_review_pipeline

    if args.text_col:
        settings["review"]["columns"]["text"] = args.text_col
    started = time.perf_counter()
    run_id, sessions = run_review_pipeline(args.csv, settings, db_path=args.db)
    elapsed = time.perf_counter() - started

    n_units = sum(len(s.units) for s in sessions)
    print(f"실행 ID: {run_id}")
    print(f"리뷰 {len(sessions)}건, 절 {n_units}개 분석 ({elapsed:.1f}초) → {args.db or settings['db_path']}")
    print()
    for name, label in INDEX_LABELS.items():
        values = [s.indices[name].raw for s in sessions if s.indices[name].raw is not None]
        if values:
            print(f"  {label:<6} 평균 {sum(values) / len(values):5.1f}  (n={len(values)})")
    print()
    print("다음: voicecx report --out reports/review_report.md")
    return 0


def cmd_report(args, settings) -> int:
    from voicecx.cx.report import build_report
    from voicecx.store import db

    conn = db.connect(args.db or settings["db_path"])
    try:
        text = build_report(conn, settings, brand=args.brand, exclude_review_event=args.exclude_review_event)
    finally:
        conn.close()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"리포트 저장: {out}")
    else:
        print(text)
    return 0


def cmd_query(args, settings) -> int:
    from voicecx.store import db

    if args.list or not args.name:
        for name in db.list_queries():
            first_line = db.read_query(name).splitlines()[0].lstrip("- ")
            print(f"{name:<28} {first_line}")
        return 0
    conn = db.connect(args.db or settings["db_path"])
    try:
        rows = [dict(r) for r in db.run_query(conn, args.name)]
    finally:
        conn.close()
    _print_table(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicecx", description="고객 감정·CX 지표 분석 도구")
    parser.add_argument("--config", help="기본 설정을 덮어쓸 YAML 파일")
    parser.add_argument("--db", help="SQLite 경로 (기본: configs/default.yaml의 db_path)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("review", help="리뷰 CSV 분석")
    p.add_argument("csv")
    p.add_argument("--text-col", help="리뷰 본문 열 이름")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("report", help="개선 과제 리포트 생성 (Markdown)")
    p.add_argument("--out", help="저장 경로. 없으면 화면에 출력")
    p.add_argument("--brand", help="특정 브랜드만")
    p.add_argument("--exclude-review-event", action="store_true", help="리뷰 이벤트 리뷰 제외")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("query", help="저장된 SQL 집계 쿼리 실행")
    p.add_argument("name", nargs="?")
    p.add_argument("--list", action="store_true")
    p.set_defaults(func=cmd_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
