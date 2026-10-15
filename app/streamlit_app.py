"""VoC Lens 대시보드.

  streamlit run app/streamlit_app.py

DB가 비어 있으면 합성 샘플 데이터로 바로 분석할 수 있다 (데모·배포용).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from voclens.config import REPO_ROOT, load_settings
from voclens.cx import compare
from voclens.cx import report as review_report
from voclens.cx.cx_report import run_cx
from voclens.cx.recovery import recovery_trigger_summary
from voclens.dashboard import charts, data
from voclens.schema import INDEX_LABELS
from voclens.store import db
from voclens.text.taxonomy import Taxonomy

PAGES = ["홈", "리뷰", "상담", "영상", "채널 비교"]
EVENT_INDEX_LABELS = {"valence": "감성", "stress": "스트레스"}
DISCLAIMER = "모든 지수는 자동 추정치입니다. 의료적 진단, 채용 평가, 직원 감시, 개인 평가에 쓸 수 없습니다."

st.set_page_config(page_title="VoC Lens", layout="wide")
settings = load_settings()


def fmt(x, digits=1, suffix=""):
    return "-" if x is None or pd.isna(x) else f"{x:.{digits}f}{suffix}"


# DB 경로 변경과 다시 분석은 공개 배포에서 막는다 (방문자가 서버 경로를 고르거나 모두의 데이터를 지우지 못하게).
# 로컬에서 쓰려면 VOCLENS_DASHBOARD_ADMIN=1 환경 변수를 켜고 실행한다.
ADMIN = os.environ.get("VOCLENS_DASHBOARD_ADMIN") == "1"
default_db = str(REPO_ROOT / settings["db_path"])

# --- 사이드바: 화면 선택 (관리 모드에서는 데이터 소스도) ---
with st.sidebar:
    st.title("VoC Lens")
    st.caption("Voice of Customer를 하나의 기준으로 보는 렌즈")
    page = st.radio("화면", PAGES, key="page")
    st.divider()
    db_path = default_db
    if ADMIN:
        db_path = st.text_input("DB 경로", value=default_db, key="db_path")
        if st.button("합성 샘플로 다시 분석", help="data/samples의 리뷰·상담·영상으로 세 채널 분석과 CX 추정을 다시 실행합니다"):
            with st.spinner("샘플 데이터 분석 중"):
                Path(db_path).unlink(missing_ok=True)
                data.build_sample_db(db_path, settings)
            st.success("샘플 분석을 마쳤습니다")
    else:
        st.caption("합성 데이터 시연입니다. 리뷰 180건, 상담 4건, 영상 2건으로 만든 결과이며 실제 고객 데이터가 아닙니다.")
        st.markdown("[GitHub 저장소](https://github.com/jhheo51-arch/VoC-Lens)")

if db_path == default_db and not Path(db_path).exists():
    # 배포 환경 첫 방문: DB가 없으면 합성 샘플로 바로 채운다
    with st.spinner("처음 실행이라 합성 샘플 데이터를 분석하고 있습니다 (몇 초 걸립니다)"):
        data.build_sample_db(db_path, settings)

conn = db.connect(db_path)
counts = data.counts(conn)

st.warning(DISCLAIMER, icon="⚠️")

if not counts:
    st.info("분석된 데이터가 없습니다. `voclens review / call / video` 명령으로 데이터를 넣거나, VOCLENS_DASHBOARD_ADMIN=1로 실행해 사이드바의 **합성 샘플로 다시 분석**을 누르세요.")
    st.stop()

if not data.has_estimates(conn):
    run_cx(conn, settings)


# =========================== 홈 ===========================
def page_home():
    st.header("한눈에 보기")
    cols = st.columns(3)
    cols[0].metric("리뷰", f"{counts.get('review', 0)}건")
    cols[1].metric("상담 통화", f"{counts.get('call', 0)}건")
    cols[2].metric("영상", f"{counts.get('video', 0)}건")

    estimates, evaluation, mentions = run_cx(conn, settings)
    summary = compare.channel_summary(estimates)
    st.subheader("채널별 CX 지표 (추정)")
    cols = st.columns(len(summary) or 1)
    for col, r in zip(cols, summary):
        with col:
            st.markdown(f"**{r['label']}** · {r['n']}건")
            st.metric("추정 CSAT% (만족 비율)", fmt(r["csat_pct"], suffix="%"))
            st.metric("추정 CES (1 쉬움~7 어려움)", fmt(r["ces_mean"], 2))
            nps = "-" if r["nps"] is None else f"{r['nps']:+.0f}"
            st.metric("NPS 성향", nps, help=f"추천·이탈 의사가 드러난 세션 {r['nps_coverage'] * 100:.0f}%로 계산")

    left, right = st.columns(2)
    with left:
        st.subheader("우선 개선 과제 (리뷰)")
        rdata = review_report.load_report_data(conn)
        pains = review_report.pain_points(rdata, Taxonomy.load(), top_n=5)
        if pains:
            st.dataframe(
                pd.DataFrame(pains)[["label", "share", "priority", "quadrant"]].assign(share=lambda d: d["share"] * 100),
                column_config={
                    "label": "불만 유형",
                    "share": st.column_config.NumberColumn("언급 비율 (%)", format="%.1f"),
                    "priority": st.column_config.ProgressColumn("우선순위", min_value=0, max_value=100, format="%.0f"),
                    "quadrant": "4분면",
                },
                hide_index=True,
                width="stretch",
            )
    with right:
        st.subheader("조용한 불만")
        gaps = compare.channel_gaps(mentions[mentions["session_id"].isin(set(estimates.loc[estimates["domain"] == "delivery", "session_id"]))])
        if gaps["silent"]:
            for g in gaps["silent"][:5]:
                flag = " ⚠ 표본 적음" if g.small_sample else ""
                st.markdown(f"- **{g.label}**: {compare.CHANNEL_LABELS[g.channel]}에서 {g.channel_share:.0f}% (리뷰 {g.review_share:.0f}%){flag}")
        else:
            st.caption("리뷰보다 다른 채널에서 크게 드러나는 불만이 없습니다.")
    if evaluation:
        st.caption(
            f"CSAT 추정 모델: 학습에 쓰지 않은 리뷰 {evaluation.n_test}건 기준 MAE {evaluation.mae_model:.3f} "
            f"(감정 점수 하나만 쓰면 {evaluation.mae_valence_only:.3f}, 평균 별점으로 찍으면 {evaluation.mae_constant:.3f})"
        )


# =========================== 리뷰 ===========================
def page_review():
    st.header("리뷰")
    brand_options = ["전체", *data.brands(conn, "review")]
    c1, c2 = st.columns([2, 1])
    brand = c1.selectbox("브랜드", brand_options, key="review_brand")
    exclude_event = c2.toggle("리뷰 이벤트 리뷰 제외", key="review_exclude_event")
    rdata = review_report.load_report_data(conn, brand=None if brand == "전체" else brand, exclude_review_event=exclude_event)
    if rdata.sessions.empty:
        st.info("조건에 맞는 리뷰가 없습니다.")
        return

    ov = review_report.overview(rdata, settings)
    cols = st.columns(len(ov["indices"]))
    for col, row in zip(cols, ov["indices"]):
        col.metric(row["name"], fmt(row["mean"]), help=f"95% 신뢰구간 {fmt(row['low'])} ~ {fmt(row['high'])}, n={row['n']}")

    pains = review_report.pain_points(rdata, Taxonomy.load())
    left, right = st.columns([3, 2])
    with left:
        if pains:
            st.altair_chart(charts.priority_quadrant(pd.DataFrame(pains)), width="stretch")
    with right:
        st.markdown("**4분면 기준**")
        st.markdown("- **Quick Win**: 우선순위 중앙값 이상 + 난이도 1\n- **핵심 과제**: 중앙값 이상 + 난이도 2~3\n- **소규모 개선**: 중앙값 미만 + 난이도 1\n- **모니터링**: 중앙값 미만 + 난이도 2~3")
        if pains:
            top = pains[0]
            st.markdown(f"**1순위: {top['label']}** ({top['quadrant']})")
            st.markdown(f"이탈 의사 {top['churn_with'] * 100:.0f}% (전체 {top['churn_all'] * 100:.0f}%)")
            for e in top["evidence"]:
                st.markdown(f"> {e}")

    share = pd.DataFrame([dict(r) for r in db.run_query(conn, "complaint_share_by_brand")])
    if not share.empty:
        if brand != "전체":
            share = share[share["brand"] == brand]
        top_labels = share.groupby("label")["share_pct"].sum().nlargest(8).index
        st.altair_chart(
            charts.grouped_share(share[share["label"].isin(top_labels)], "label", "brand", "share_pct", "브랜드별 불만 유형 언급 비율"),
            width="stretch",
        )

    left, right = st.columns(2)
    journey = pd.DataFrame(review_report.journey_table(rdata, settings["report"]["min_n"]))
    aspects = pd.DataFrame(review_report.aspect_table(rdata))
    with left:
        if not journey.empty:
            st.altair_chart(charts.diverging_bars(journey, "stage", "valence", "고객 여정 단계별 감성", "n"), width="stretch")
            small = journey.loc[journey["small"], "stage"].tolist()
            if small:
                st.caption(f"표본 {settings['report']['min_n']}개 미만: {', '.join(small)} (해석 주의)")
    with right:
        if not aspects.empty:
            st.altair_chart(charts.diverging_bars(aspects, "aspect", "sentiment", "속성별 감성", "n"), width="stretch")

    mism = review_report.mismatches(rdata)
    with st.expander(f"별점–감정 불일치 리뷰 ({len(mism)}건)"):
        if mism:
            st.dataframe(pd.DataFrame(mism), hide_index=True, width="stretch")
        else:
            st.caption("별점과 감정 방향이 반대인 리뷰가 없습니다.")


# =========================== 상담 ===========================
def page_call():
    st.header("상담")
    calls = data.call_list(conn)
    if calls.empty:
        st.info("분석된 상담이 없습니다. `voclens call <전사본 폴더>`로 넣어 주세요.")
        return
    st.dataframe(
        calls[["call_id", "outcome", "frustration", "recovery", "peak_end", "ces_est", "customer_interruptions", "repeat_count", "max_silence"]],
        column_config={
            "call_id": "통화", "outcome": "결과",
            "frustration": st.column_config.ProgressColumn("좌절", min_value=0, max_value=100, format="%.1f"),
            "recovery": st.column_config.NumberColumn("감정 회복", format="%.1f"),
            "peak_end": st.column_config.NumberColumn("Peak-End", format="%.1f"),
            "ces_est": st.column_config.NumberColumn("추정 CES", format="%.2f"),
            "customer_interruptions": "끼어들기", "repeat_count": "반복", "max_silence": st.column_config.NumberColumn("최장 침묵 (초)", format="%.1f"),
        },
        hide_index=True,
        width="stretch",
    )
    call_id = st.selectbox("통화 선택", calls["call_id"], key="call_select")
    row = calls[calls["call_id"] == call_id].iloc[0]
    if row.get("scenario"):
        st.caption(f"시나리오: {row['scenario']}")
    units = data.session_units(conn, row["session_id"])
    events = data.session_events(conn, row["session_id"])
    st.altair_chart(charts.emotion_timeline(units, events), width="stretch")
    if not events.empty:
        for e in events.itertuples():
            direction = "회복" if e.delta > 0 else "악화"
            trigger = f" · 직전 상담원: “{e.trigger_text}”" if e.trigger_text else ""
            name = EVENT_INDEX_LABELS.get(e.index_name, e.index_name)
            if e.index_name == "stress":
                direction = "상승" if e.delta > 0 else "하락"
            st.markdown(f"- {e.start_sec:.0f}~{e.end_sec:.0f}초 고객 {name} {direction} ({e.delta:+.2f}){trigger}")

    with st.expander("전사본"):
        st.dataframe(
            units[["start_sec", "role", "text", "valence", "complaints", "signals"]],
            column_config={"start_sec": st.column_config.NumberColumn("시작 (초)", format="%.1f"), "role": "화자", "text": "발화",
                           "valence": st.column_config.NumberColumn("감성", format="%+.2f"), "complaints": "불만 유형", "signals": "CX 신호"},
            hide_index=True,
            width="stretch",
        )

    triggers = pd.DataFrame(recovery_trigger_summary(conn))
    if not triggers.empty and triggers["n_trigger"].sum() > 0:
        st.altair_chart(charts.lift_bars(triggers.dropna(subset=["lift"])), width="stretch")
        st.caption("lift = 회복 직전 상담원 발화에서의 비율 / 전체 상담원 발화에서의 비율. 연관이며 인과가 아닙니다. 건수가 적으면 우연일 수 있습니다.")


# =========================== 영상 ===========================
def page_video():
    st.header("영상")
    videos = data.video_list(conn)
    if videos.empty:
        st.info("분석된 영상이 없습니다. `voclens video <번들 폴더 또는 URL>`로 넣어 주세요.")
        return
    labels = {r.session_id: f"{r.title or r.video_id} ({r.video_type})" for r in videos.itertuples()}
    session_id = st.selectbox("영상 선택", list(labels), format_func=labels.get, key="video_select")
    meta = data.session_meta(conn, session_id)
    idx = data.session_indices(conn, session_id)
    cva = meta.get("creator_vs_audience", {})

    cols = st.columns(4)
    cols[0].metric("총평 가중 감성", fmt(meta.get("weighted_valence"), 2))
    cols[1].metric("행복", fmt(idx.get("happiness")))
    cols[2].metric("몰입 (시청자 반응)", fmt(idx.get("engagement")))
    cols[3].metric("시청자 − 크리에이터 감성", fmt(cva.get("audience_minus_creator"), 2), help="양수면 댓글이 크리에이터보다 긍정적")

    units = data.session_units(conn, session_id)
    segments = data.video_segments(meta)
    if not segments.empty:
        segments["comment_valence"] = pd.to_numeric(segments["comment_valence"], errors="coerce")
    st.altair_chart(charts.video_timeline(units, segments), width="stretch")
    st.caption("파란 선·점 = 크리에이터 문장 감성, 주황 점 = 게스트, 청록 마름모 = 타임스탬프 댓글(가리킨 시각), 회색 띠 = 내용 구간")

    if not segments.empty:
        st.dataframe(
            segments[["name", "start_sec", "end_sec", "n_sentences", "valence", "aspects", "n_timestamp_comments", "comment_valence"]],
            column_config={"name": "구간", "start_sec": st.column_config.NumberColumn("시작 (초)", format="%.0f"),
                           "end_sec": st.column_config.NumberColumn("끝 (초)", format="%.0f"), "n_sentences": "문장",
                           "valence": st.column_config.NumberColumn("감성", format="%+.2f"), "aspects": "주요 속성",
                           "n_timestamp_comments": "타임스탬프 댓글", "comment_valence": st.column_config.NumberColumn("댓글 감성", format="%+.2f")},
            hide_index=True,
            width="stretch",
        )
    comments = units[units["speaker_role"] == "commenter"]
    with st.expander(f"댓글 ({len(comments)}개)"):
        st.dataframe(
            comments[["start_sec", "segment", "text", "valence", "complaints"]],
            column_config={"start_sec": st.column_config.NumberColumn("가리킨 시각 (초)", format="%.0f"), "segment": "구간",
                           "text": "댓글", "valence": st.column_config.NumberColumn("감성", format="%+.2f"), "complaints": "불만 유형"},
            hide_index=True,
            width="stretch",
        )


# =========================== 채널 비교 ===========================
def page_compare():
    st.header("채널 비교")
    domain_label = st.segmented_control("도메인", ["배달", "제품", "전체"], default="배달", key="compare_domain")
    domain = {"배달": "delivery", "제품": "product", "전체": None}[domain_label or "배달"]
    estimates, evaluation, mentions = run_cx(conn, settings, domain=domain)

    summary = pd.DataFrame(compare.channel_summary(estimates))
    if not summary.empty:
        summary["actual_csat_pct"] = pd.to_numeric(summary["actual_csat_pct"], errors="coerce")
        st.dataframe(
            summary[["label", "n", "csat_est_mean", "csat_pct", "actual_csat_pct", "ces_mean", "nps", "nps_coverage", "happiness", "frustration"]],
            column_config={"label": "채널", "n": "세션",
                           "csat_est_mean": st.column_config.NumberColumn("추정 CSAT", format="%.2f"),
                           "csat_pct": st.column_config.NumberColumn("추정 CSAT%", format="%.1f"),
                           "actual_csat_pct": st.column_config.NumberColumn("실제 CSAT% (별점)", format="%.1f"),
                           "ces_mean": st.column_config.NumberColumn("추정 CES", format="%.2f"),
                           "nps": st.column_config.NumberColumn("NPS 성향", format="%+.0f"),
                           "nps_coverage": st.column_config.NumberColumn("판단 가능 비율", format="percent"),
                           "happiness": st.column_config.NumberColumn("행복", format="%.1f"),
                           "frustration": st.column_config.NumberColumn("좌절", format="%.1f")},
            hide_index=True,
            width="stretch",
        )

    mix = compare.complaint_mix(mentions)
    if mix.empty:
        st.info("이 도메인에는 불만 언급이 없습니다.")
        return
    left, right = st.columns([3, 2])
    with left:
        top = mix.sum(axis=0).nlargest(12).index
        st.altair_chart(charts.share_heatmap(mix[top], compare.CHANNEL_LABELS), width="stretch")
        with st.expander("표로 보기"):
            st.dataframe(mix[top].T.rename(columns=compare.CHANNEL_LABELS).round(1), width="stretch")
    with right:
        test = compare.mix_test(mentions)
        if test:
            verdict = "채널마다 불만 구성이 다릅니다" if test.significant else "채널 간 차이가 유의하지 않습니다"
            st.markdown(f"**{verdict}**")
            st.caption(f"카이제곱 p = {test.p_value:.4f}, Cramér's V = {test.effect_size:.2f} (효과 크기 {test.effect_label})")
        gaps = compare.channel_gaps(mentions)
        st.markdown("**조용한 불만** (리뷰보다 다른 채널에서 비중이 큰 유형)")
        if gaps["silent"]:
            st.dataframe(
                pd.DataFrame([
                    {"불만 유형": g.label, "채널": compare.CHANNEL_LABELS[g.channel], "채널 비중 (%)": round(g.channel_share, 1),
                     "리뷰 비중 (%)": round(g.review_share, 1), "표본": "⚠ 적음" if g.small_sample else f"{g.channel_total}건"}
                    for g in gaps["silent"]
                ]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("해당 유형이 없습니다.")
        st.markdown("**리뷰에서만 큰 불만**")
        if gaps["review_only"]:
            st.markdown("\n".join(f"- {g.label}: 리뷰 {g.review_share:.0f}% / 다른 채널 합계 {g.channel_share:.0f}%" for g in gaps["review_only"]))
        else:
            st.caption("해당 유형이 없습니다.")

    if evaluation:
        with st.expander("CSAT 추정 모델 검증"):
            st.dataframe(
                pd.DataFrame(
                    {"모델": ["평균 별점으로 찍기", "감정 점수 하나만", "전체 특징 릿지 회귀"],
                     "MAE": [evaluation.mae_constant, evaluation.mae_valence_only, evaluation.mae_model]}
                ).round(3),
                hide_index=True,
            )
            st.caption(
                f"평가 리뷰 {evaluation.n_test}건, Spearman ρ {evaluation.spearman:.2f}. 만족 비율: 실제 {evaluation.csat_pct_actual:.1f}% / "
                f"추정 {evaluation.csat_pct_est:.1f}% (보정 전 {evaluation.csat_pct_est_raw:.1f}%)"
            )


{"홈": page_home, "리뷰": page_review, "상담": page_call, "영상": page_video, "채널 비교": page_compare}[page]()
conn.close()
