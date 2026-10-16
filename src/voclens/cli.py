"""voclens 명령줄 도구.

  voclens review data/samples/delivery_reviews_synthetic.csv
  voclens call data/samples/calls
  voclens video data/samples/videos
  voclens report --out reports/review_report.md
  voclens cx --out reports/cx_report.md
  voclens query --list
  voclens query complaint_share_by_brand
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from voclens.config import load_settings
from voclens.schema import INDEX_LABELS


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
    from voclens.pipelines.review import run_review_pipeline

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
    print("다음: voclens report --out reports/review_report.md")
    return 0


OUTCOME_LABELS = {"resolved": "해결", "pending": "보류", "transferred": "이관", "unknown": "불명"}


def cmd_call(args, settings) -> int:
    from voclens.cx.recovery import recovery_trigger_summary
    from voclens.pipelines.call import run_call_pipeline
    from voclens.store import db

    started = time.perf_counter()
    run_id, sessions = run_call_pipeline(args.paths, settings, db_path=args.db)
    elapsed = time.perf_counter() - started
    print(f"실행 ID: {run_id}")
    print(f"통화 {len(sessions)}건 분석 ({elapsed:.1f}초) → {args.db or settings['db_path']}")
    print()
    rows = []
    for s in sessions:
        m = s.meta
        rows.append(
            {
                "통화": s.source_id,
                "결과": OUTCOME_LABELS.get(m["outcome"], m["outcome"]),
                "좌절": s.indices["frustration"].raw,
                "스트레스": s.indices["stress"].raw,
                "감정 회복": s.indices["recovery"].raw,
                "Peak-End": s.peak_end,
                "끼어들기": m["customer_interruptions"],
                "반복": m["repeat_count"],
                "최장 침묵": m["max_silence"],
                "음성": "사용" if m.get("audio_used") else "-",
                "역할 신뢰도": m["role_confidence"],
            }
        )
    _print_table(rows)

    conn = db.connect(args.db or settings["db_path"])
    try:
        triggers = recovery_trigger_summary(conn)
    finally:
        conn.close()
    if any(t["n_trigger"] for t in triggers):
        print()
        print("회복 직전 상담원 표현 (DB에 쌓인 전체 통화 기준, lift > 1이면 회복 전에 자주 나옴)")
        _print_table(triggers)
    return 0


def cmd_video(args, settings) -> int:
    from voclens.pipelines.video import run_video_pipeline

    started = time.perf_counter()
    run_id, sessions = run_video_pipeline(args.sources, settings, db_path=args.db, allow_stt=args.stt)
    elapsed = time.perf_counter() - started
    print(f"실행 ID: {run_id}")
    print(f"영상 {len(sessions)}건 분석 ({elapsed:.1f}초) → {args.db or settings['db_path']}")
    for s in sessions:
        m = s.meta
        cva = m["creator_vs_audience"]
        print()
        print(f"[{s.source_id}] {m.get('title') or ''}")
        print(f"  유형 {m['video_type_label']} (신뢰도 {m['video_type_confidence']}), 자막 출처 {m['transcript_source']}, "
              f"문장 {m['n_sentences']}개, 댓글 {m['n_comments']}개 (타임스탬프 {m['n_timestamp_comments']}개)")
        print(f"  행복 {s.indices['happiness'].raw}  스트레스 {s.indices['stress'].raw}  몰입 {s.indices['engagement'].raw}  "
              f"총평 가중 Valence {m['weighted_valence']}  시청자-크리에이터 Valence 차이 {cva['audience_minus_creator']}")
        _print_table(
            [
                {
                    "구간": f"{seg['index']} {seg['label']}",
                    "시작": seg["start_sec"],
                    "끝": seg["end_sec"],
                    "문장": seg["n_sentences"],
                    "Valence": seg["valence"],
                    "속성": ", ".join(seg["aspects"]),
                    "타임스탬프 댓글": seg["n_timestamp_comments"],
                    "댓글 Valence": seg["comment_valence"],
                }
                for seg in m["segments"]
            ]
        )
    return 0


def cmd_cx(args, settings) -> int:
    from voclens.cx.cx_report import build_cx_report
    from voclens.store import db

    conn = db.connect(args.db or settings["db_path"])
    try:
        text = build_cx_report(conn, settings, brand=args.brand, domain=args.domain)
    finally:
        conn.close()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"CX 리포트 저장: {out}")
    else:
        print(text)
    return 0


def cmd_label(args, settings) -> int:
    from voclens import labeling

    if args.action == "prepare":
        if args.text_col:
            settings["review"]["columns"]["text"] = args.text_col
        table = labeling.clause_table(args.input, settings, domain=args.domain)
        encode = None
        if args.embed:
            from voclens.text.taxonomy import sentence_transformer_encoder

            encode = sentence_transformer_encoder()
        picked = labeling.select_for_labeling(labeling.score_uncertainty(table, encode), n=args.n, per_review=args.per_review)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(labeling.to_label_studio(picked), ensure_ascii=False, indent=1), encoding="utf-8")
        basis = "임베딩 1·2위 유사도 차이" if args.embed else "규칙 기반 대리 지표"
        print(f"절 {len(table):,}개 중 헷갈리는 순서로 {len(picked)}개 → {out} (기준: {basis})")
        if "kind" in picked:
            print(picked["kind"].value_counts().to_string())
        print("Label Studio에서 labeling/label_studio_config.xml로 프로젝트를 만들고 이 파일을 가져오세요.")
        return 0

    if args.action == "import":
        df = labeling.read_label_studio_export(args.input)
        out = labeling.labels_to_csv(df, args.out, with_text=args.with_text)
        print(f"라벨 {len(df)}개 (보류 {int(df['hold'].sum())}개) → {out}")
        return 0

    first = labeling.read_label_studio_export(args.input)
    second = labeling.read_label_studio_export(args.second)
    result = labeling.agreement(first, second)
    print(f"공통 절 {result['n_common']}개 · 완전 일치 {result['exact_match']:.1%} · macro κ {result['macro_kappa']}")
    _print_table([{"유형": k, "κ": v} for k, v in sorted(result["per_label"].items(), key=lambda kv: (kv[1] is None, kv[1]))])
    if result["macro_kappa"] is not None and result["macro_kappa"] < 0.7:
        print("목표(0.7)보다 낮습니다. 일치하지 않은 절을 labeling/guidelines.md 판정 사례에 추가하고 기준을 다듬으세요.")
    return 0


def cmd_eval(args, settings) -> int:
    import pandas as pd

    from voclens import evaluation as ev
    from voclens import labeling

    out = Path(args.out) if args.out else None
    if args.action == "kote":
        df = ev.read_kote(args.input)
        result = ev.evaluate_emotions(df["text"].tolist(), df["labels"].tolist())
        pol = result["polarity"]
        print(f"KOTE {result['n']:,}문장 · 감정 표현 검출 {result['coverage']:.1%}")
        print(f"극성: AUC {pol['auc']} · 판단한 비율 {pol['called_share']:.1%} · 판단했을 때 정확도 {pol['accuracy_when_called']:.1%}"
              f" (긍정 {pol['accuracy_positive']:.1%}, 부정 {pol['accuracy_negative']:.1%})")
        _print_table([{"감정": k, **v} for k, v in result["per_emotion"].items()])
        print(f"감정 macro F1 {result['macro_f1']} · '없음' 문장에 감정이 걸린 비율 {result['none_false_hit']:.1%}")
    elif args.action == "ratings":
        df = ev.read_shopping(args.input, args.split)
        if args.n:
            df = pd.concat([g.sample(min(args.n, len(g)), random_state=0) for _, g in df.groupby("rating")])
        result = ev.evaluate_ratings(df, settings, domain=args.domain)
        print(f"리뷰 {result['n']:,}건 ({args.split or '전체'}) · 분류 기준 v{result['taxonomy_version']}")
        print(f"리뷰 평균 Valence로 4~5점 vs 1~2점 구분 AUC {result['auc_valence_vs_rating']}")
        _print_table([{"별점": k, "불만 유형이 붙은 리뷰": f"{v:.1%}"} for k, v in result["complaint_rate_by_rating"].items()])
        _print_table([{"불만 유형 (1~2점 리뷰)": k, "비율": f"{v:.1%}"} for k, v in list(result["negative_review_mix"].items())[:15]])
    elif args.action == "prepare":
        df = ev.sample_eval_clauses(ev.read_shopping(args.input, "test"), per_rating=args.per_rating)
        target = out or Path("data/labeling/eval_tasks.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ev.to_eval_tasks(df), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"평가용 리뷰에서 무작위로 절 {len(df)}개 → {target} (별점별 {args.per_rating}건, 모델 예측은 숨김)")
        print("원문이 들어 있으니 저장소에 올리지 마세요 (data/ 는 .gitignore).")
        return 0
    else:
        gold = labeling.read_label_studio_export(args.input)
        shopping = ev.read_shopping(args.shopping)
        result = ev.evaluate_gold(gold, ev.predict_units(gold["unit_id"].tolist(), shopping, settings, args.domain))
        a, m = result["any_complaint"], result["micro"]
        print(f"평가 절 {result['n']}개 (보류 {result['held']}개 제외)")
        print(f"불만 여부: 정밀도 {a['precision']} · 재현율 {a['recall']} · F1 {a['f1']}")
        print(f"불만 유형: micro F1 {m['f1']} · macro F1 {result['macro_f1']}")
        _print_table([{"유형": k, **v} for k, v in sorted(result["per_label"].items(), key=lambda kv: -kv[1]["support"])])
    if out:
        print(f"결과 저장 → {ev.save_result(result, out)}")
    return 0


def cmd_report(args, settings) -> int:
    from voclens.cx.report import build_report
    from voclens.store import db

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
    from voclens.store import db

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
    parser = argparse.ArgumentParser(prog="voclens", description="고객 감정·CX 지표 분석 도구")
    parser.add_argument("--config", help="기본 설정을 덮어쓸 YAML 파일")
    parser.add_argument("--db", help="SQLite 경로 (기본: configs/default.yaml의 db_path)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("review", help="리뷰 CSV 분석")
    p.add_argument("csv")
    p.add_argument("--text-col", help="리뷰 본문 열 이름")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("call", help="상담 전사본(JSON)·음성 파일 분석")
    p.add_argument("paths", nargs="+", help="전사본 JSON, 음성 파일, 또는 JSON이 든 폴더")
    p.set_defaults(func=cmd_call)

    p = sub.add_parser("video", help="영상 번들 JSON 또는 YouTube URL 분석")
    p.add_argument("sources", nargs="+", help="번들 JSON, 번들이 든 폴더, 또는 YouTube URL")
    p.add_argument("--stt", action="store_true", help="사람 자막이 없으면 음성을 받아 전사 (선택 의존성 필요)")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("cx", help="채널 통합 CX 리포트: CSAT·CES·NPS 추정, 채널 비교 (Markdown)")
    p.add_argument("--out", help="저장 경로. 없으면 화면에 출력")
    p.add_argument("--brand", help="특정 브랜드만")
    p.add_argument("--domain", default="default", help="비교할 도메인: delivery | product | all (기본: 설정의 리뷰 도메인)")
    p.set_defaults(func=cmd_cx)

    p = sub.add_parser("report", help="개선 과제 리포트 생성 (Markdown)")
    p.add_argument("--out", help="저장 경로. 없으면 화면에 출력")
    p.add_argument("--brand", help="특정 브랜드만")
    p.add_argument("--exclude-review-event", action="store_true", help="리뷰 이벤트 리뷰 제외")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("label", help="라벨링 준비·가져오기·자기 일치도 (2주차)")
    p.add_argument("action", choices=["prepare", "import", "agreement"])
    p.add_argument("input", help="prepare: 리뷰 CSV / import·agreement: Label Studio 내보내기 JSON")
    p.add_argument("second", nargs="?", help="agreement: 두 번째 라벨링 내보내기 JSON")
    p.add_argument("--out", default="labeling/tasks.json", help="prepare: 작업 JSON, import: 라벨 CSV 경로")
    p.add_argument("--n", type=int, default=500, help="prepare: 고를 절 수")
    p.add_argument("--per-review", type=int, default=2, help="prepare: 리뷰 하나에서 고를 최대 절 수")
    p.add_argument("--domain", help="prepare: delivery | product")
    p.add_argument("--text-col", help="prepare: 리뷰 본문 열 이름")
    p.add_argument("--embed", action="store_true", help="prepare: 임베딩 Zero-shot 유사도 차이로 고름 (pip install -e .[nlp])")
    p.add_argument("--with-text", action="store_true", help="import: CSV에 원문도 넣음 (외부 데이터는 넣지 말 것)")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("eval", help="공개 실제 데이터로 기준선 평가: KOTE 감정, 네이버 쇼핑 별점, 직접 라벨링한 평가 세트")
    p.add_argument("action", choices=["kote", "ratings", "prepare", "gold"])
    p.add_argument("input", help="kote: KOTE tsv / ratings·prepare: naver_shopping.txt / gold: Label Studio 내보내기 JSON")
    p.add_argument("--shopping", default="data/external/naver_shopping/naver_shopping.txt", help="gold: naver_shopping.txt 경로")
    p.add_argument("--split", choices=["dev", "test"], help="ratings: 개발용(80%%) 또는 평가용(20%%)만")
    p.add_argument("--n", type=int, help="ratings: 별점마다 무작위로 n건만")
    p.add_argument("--per-rating", type=int, default=50, help="prepare: 별점마다 뽑을 리뷰 수 (리뷰당 절 2개까지)")
    p.add_argument("--domain", default="product", help="ratings·gold: 분류 기준 도메인")
    p.add_argument("--out", help="결과 JSON 경로 (원문 없이 지표만). prepare: 작업 JSON 경로")
    p.set_defaults(func=cmd_eval)

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
