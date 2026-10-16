"""VoC Lens 대시보드.

  streamlit run app/streamlit_app.py

- 샘플 화면(홈·리뷰·상담·영상·채널 비교): 합성 샘플 DB. 없으면 첫 실행 때 자동으로 만든다.
- 직접 분석: 방문자가 올린 리뷰·상담·영상을 방문자 전용 메모리 DB에서 분석한다 (저장하지 않음).
- 분석 기준: 설정 파일에서 바로 만든 규칙 표.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from voclens.config import REPO_ROOT, load_settings
from voclens.cx import compare
from voclens.cx import report as review_report
from voclens.cx.cx_report import run_cx
from voclens.cx.recovery import recovery_trigger_summary
from voclens.dashboard import charts, criteria, data, upload
from voclens.dashboard.evidence import add_evidence
from voclens.store import db
from voclens.text.taxonomy import Taxonomy

PAGES = ["홈", "직접 분석", "분석 기준"]
# 홈 안의 탭: 직접 분석을 하면 나오는 결과 화면을 합성 데이터로 미리 보여준다
HOME_TABS = ["요약", "리뷰", "상담", "영상", "채널 비교"]
EVENT_INDEX_LABELS = {"valence": "감성", "stress": "스트레스"}
DOMAINS = {"배달": "delivery", "온라인 쇼핑·제품": "product"}
DISCLAIMER = "모든 지수는 자동 추정치입니다. 의료적 진단, 채용 평가, 직원 감시, 개인 평가에 쓸 수 없습니다."

st.set_page_config(page_title="VoC Lens", layout="wide")
settings = load_settings()

# DB 경로 변경과 다시 분석은 공개 배포에서 막는다 (방문자가 서버 경로를 고르거나 모두의 데이터를 지우지 못하게).
# 로컬에서 쓰려면 VOCLENS_DASHBOARD_ADMIN=1 환경 변수를 켜고 실행한다.
ADMIN = os.environ.get("VOCLENS_DASHBOARD_ADMIN") == "1"
default_db = str(REPO_ROOT / settings["db_path"])


def fmt(x, digits=1, suffix=""):
    return "-" if x is None or pd.isna(x) else f"{x:.{digits}f}{suffix}"


def criteria_box(*sections: str):
    """화면 아래 '이 화면의 분석 기준' 접이식 상자. sections: review | call | video | cx | emotion"""
    builders = {
        "emotion": ("감정·지수", criteria.emotion_rules),
        "review": ("리뷰", criteria.review_rules),
        "call": ("상담", criteria.call_rules),
        "video": ("영상", criteria.video_rules),
        "cx": ("CX 지표", criteria.cx_rules),
    }
    with st.expander("이 화면의 분석 기준"):
        for key in sections:
            title, fn = builders[key]
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {line}" for line in fn()))
        st.caption("전체 기준과 사전·분류표는 사이드바의 **분석 기준** 화면에 있습니다.")


EVIDENCE_COLUMNS = {
    "text": "문장",
    "valence": st.column_config.NumberColumn("감성", format="%+.2f"),
    "complaints": "불만 유형",
    "감정 근거": st.column_config.TextColumn("감정 근거", help="사전에서 걸린 표현과 감정. 강조·부정·가정은 괄호에 표시"),
    "불만 근거": st.column_config.TextColumn("불만 근거", help="불만 표현은 감정과 상관없이, 주제 표현은 감성이 음수일 때만 불만으로 판단"),
    "신호 근거": st.column_config.TextColumn("신호 근거", help="재주문·이탈·추천 의사, 반복 문의, 대기 등"),
}


# --- 사이드바 ---
with st.sidebar:
    st.title("VoC Lens")
    st.caption("Voice of Customer를 하나의 기준으로 보는 렌즈")
    page = st.radio("화면", PAGES, key="page", captions=["예시 결과 (합성 데이터)", "내 데이터 올려서 분석", "규칙·사전·공식"])
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
        st.caption("홈은 합성 데이터(리뷰 180건, 상담 4건, 영상 2건)로 만든 예시 결과입니다. 내 데이터는 **직접 분석**에서 올리면 같은 화면으로 나옵니다.")
        st.markdown("[GitHub 저장소](https://github.com/jhheo51-arch/VoC-Lens)")

st.warning(DISCLAIMER, icon="⚠️")


def sample_conn():
    if db_path == default_db and not Path(db_path).exists():
        # 배포 환경 첫 방문: DB가 없으면 합성 샘플로 바로 채운다
        with st.spinner("처음 실행이라 합성 샘플 데이터를 분석하고 있습니다 (몇 초 걸립니다)"):
            data.build_sample_db(db_path, settings)
    conn = db.connect(db_path)
    if data.counts(conn) and not data.has_estimates(conn):
        run_cx(conn, settings)
    return conn


# =========================== 홈 ===========================
def page_home(conn):
    counts = data.counts(conn)
    st.subheader("한눈에 보기")
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
    criteria_box("emotion", "cx")


# =========================== 리뷰 ===========================
def page_review(conn, k: str = "", domain: str | None = "delivery", show_brand_filter: bool = True):
    brand = "전체"
    exclude_event = False
    c1, c2 = st.columns([2, 1])
    if show_brand_filter:
        brand = c1.selectbox("브랜드", ["전체", *data.brands(conn, "review")], key=f"{k}review_brand")
    exclude_event = c2.toggle("리뷰 이벤트 리뷰 제외", key=f"{k}review_exclude_event")
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
        else:
            st.caption("불만 유형이 붙은 절이 없습니다.")
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
            st.caption("별점과 감정 방향이 반대인 리뷰가 없거나, 별점이 있는 리뷰가 10건 미만입니다.")

    with st.expander("문장별 판단 근거"):
        units = rdata.units.merge(
            rdata.labels.groupby("unit_id")["label"].apply(", ".join).rename("complaints"), on="unit_id", how="left"
        )
        only_complaints = st.toggle("불만이 붙은 절만", value=True, key=f"{k}review_evidence_filter")
        if only_complaints:
            units = units[units["complaints"].notna() & ~units["complaints"].isin(["칭찬"])]
        shown = add_evidence(units.head(300), domain)
        st.dataframe(shown[["text", "valence", "complaints", "감정 근거", "불만 근거", "신호 근거"]], column_config=EVIDENCE_COLUMNS, hide_index=True, width="stretch")
        st.caption(f"최대 300개 절을 보여줍니다 (전체 {len(units)}개).")
    criteria_box("review", "emotion")


# =========================== 상담 ===========================
def page_call(conn, k: str = "", domain: str | None = "delivery"):
    calls = data.call_list(conn)
    if calls.empty:
        st.info("분석된 상담이 없습니다.")
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
    call_id = st.selectbox("통화 선택", calls["call_id"], key=f"{k}call_select")
    row = calls[calls["call_id"] == call_id].iloc[0]
    if row.get("scenario"):
        st.caption(f"시나리오: {row['scenario']}")
    meta = data.session_meta(conn, row["session_id"])
    roles = meta.get("speaker_roles", {})
    if roles:
        method = {"given": "전사본에 적힌 역할", "rules": "정형 표현 규칙", "single": "화자 1명"}.get(meta.get("role_method"), meta.get("role_method"))
        st.caption(f"화자 역할: {', '.join(f'{s} = {data.ROLE_LABELS.get(r, r)}' for s, r in roles.items())} (판정 방법: {method}, 신뢰도 {meta.get('role_confidence')}). 음성 분석: {'사용' if meta.get('audio_used') else '안 함 (음성 파일 없음)'}")
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

    with st.expander("전사본과 판단 근거"):
        shown = add_evidence(units, domain)
        st.dataframe(
            shown[["start_sec", "role", "text", "valence", "complaints", "signals", "감정 근거", "불만 근거"]],
            column_config={**EVIDENCE_COLUMNS, "start_sec": st.column_config.NumberColumn("시작 (초)", format="%.1f"), "role": "화자",
                           "text": "발화", "signals": "CX 신호 (대화 흐름 포함)"},
            hide_index=True,
            width="stretch",
        )
        st.caption("불만 유형과 CX 신호는 고객 발화에만 붙입니다. '반복 문의'와 '대기'는 사전이 아니라 대화 흐름(같은 말 반복, 직전 침묵)으로도 붙습니다.")

    triggers = pd.DataFrame(recovery_trigger_summary(conn))
    if not triggers.empty and triggers["n_trigger"].sum() > 0:
        st.altair_chart(charts.lift_bars(triggers.dropna(subset=["lift"])), width="stretch")
        st.caption("lift = 회복 직전 상담원 발화에서의 비율 / 전체 상담원 발화에서의 비율. 연관이며 인과가 아닙니다. 건수가 적으면 우연일 수 있습니다.")
    criteria_box("call", "emotion")


# =========================== 영상 ===========================
def page_video(conn, k: str = ""):
    videos = data.video_list(conn)
    if videos.empty:
        st.info("분석된 영상이 없습니다.")
        return
    labels = {r.session_id: f"{r.title or r.video_id} ({r.video_type})" for r in videos.itertuples()}
    session_id = st.selectbox("영상 선택", list(labels), format_func=labels.get, key=f"{k}video_select")
    meta = data.session_meta(conn, session_id)
    idx = data.session_indices(conn, session_id)
    cva = meta.get("creator_vs_audience", {})

    cols = st.columns(4)
    cols[0].metric("총평 가중 감성", fmt(meta.get("weighted_valence"), 2))
    cols[1].metric("행복", fmt(idx.get("happiness")))
    engagement_label = "몰입 (시청자 반응)" if meta.get("engagement_basis") == "audience" else "몰입 (자막 내용)"
    cols[2].metric(engagement_label, fmt(idx.get("engagement")))
    cols[3].metric("시청자 − 크리에이터 감성", fmt(cva.get("audience_minus_creator"), 2), help="양수면 댓글이 크리에이터보다 긍정적")
    st.caption(f"영상 유형: {meta.get('video_type_label')} (신뢰도 {meta.get('video_type_confidence')}) · 자막 출처: {meta.get('transcript_source')} · 문장 {meta.get('n_sentences')}개 · 댓글 {meta.get('n_comments')}개")

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
    domain = meta.get("domain")
    with st.expander("문장별 판단 근거 (자막)"):
        speech = add_evidence(units[units["speaker_role"] != "commenter"], domain)
        st.dataframe(speech[["start_sec", "segment", "role", "text", "valence", "complaints", "감정 근거", "불만 근거"]],
                     column_config={**EVIDENCE_COLUMNS, "start_sec": st.column_config.NumberColumn("시작 (초)", format="%.0f"), "segment": "구간", "role": "화자"},
                     hide_index=True, width="stretch")
    comments = units[units["speaker_role"] == "commenter"]
    with st.expander(f"댓글과 판단 근거 ({len(comments)}개)"):
        shown = add_evidence(comments, domain)
        st.dataframe(
            shown[["start_sec", "segment", "text", "valence", "complaints", "감정 근거", "불만 근거"]],
            column_config={**EVIDENCE_COLUMNS, "start_sec": st.column_config.NumberColumn("가리킨 시각 (초)", format="%.0f"), "segment": "구간", "text": "댓글"},
            hide_index=True,
            width="stretch",
        )
    criteria_box("video", "emotion")


# =========================== 채널 비교 ===========================
def page_compare(conn):
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
    criteria_box("cx")


# =========================== 직접 분석 ===========================
SAMPLES = REPO_ROOT / "data" / "samples"


def _sample_call_text() -> str:
    call = json.loads((SAMPLES / "calls" / "call_002_late_delivery.json").read_text(encoding="utf-8"))
    names = {"A": "상담원", "B": "고객"}
    return "\n".join(f"[{int(t['start'] // 60):02d}:{int(t['start'] % 60):02d}] {names[t['speaker']]}: {t['text']}" for t in call["turns"])


def _sample_vtt() -> bytes:
    bundle = json.loads((SAMPLES / "videos" / "video_001_vacuum_review.json").read_text(encoding="utf-8"))

    def ts(sec):
        return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:06.3f}"

    cues = [f"{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}" for s in bundle["segments"]]
    return ("WEBVTT\n\n" + "\n\n".join(cues)).encode("utf-8")


def _sample_comments() -> str:
    bundle = json.loads((SAMPLES / "videos" / "video_001_vacuum_review.json").read_text(encoding="utf-8"))
    return "\n".join(c["text"] for c in bundle["comments"])


def _show_notes(notes):
    for n in notes:
        st.info(n)


def upload_review_tab():
    st.markdown("CSV 파일을 올리면 리뷰마다 감정·불만 유형·CX 신호를 붙이고 개선 과제를 정리합니다. 본문 열은 꼭 있어야 하고, 별점·작성일·브랜드 열은 있으면 씁니다.")
    file = st.file_uploader("리뷰 CSV", type=["csv", "tsv"], key="up_review_file")
    use_sample = st.button("합성 샘플 CSV로 해 보기", key="up_review_sample")
    if use_sample:
        st.session_state["up_review_bytes"] = ((SAMPLES / "delivery_reviews_synthetic.csv").read_bytes(), "delivery_reviews_synthetic.csv")
    if file is not None:
        st.session_state["up_review_bytes"] = (file.getvalue(), file.name)
    if "up_review_bytes" not in st.session_state:
        return
    raw, name = st.session_state["up_review_bytes"]
    try:
        df = upload.read_table(raw, name)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"{name} · {len(df):,}행 · 열: {', '.join(df.columns)}")
    cols = list(df.columns)
    none = "(없음)"

    def pick(label, candidates, key):
        guess = next((c for c in cols if c.strip().lower() in candidates), none)
        return st.selectbox(label, [none, *cols], index=([none, *cols].index(guess)), key=key)

    c = st.columns(5)
    with c[0]:
        text_col = st.selectbox("본문 열", cols, index=cols.index(upload.guess_text_column(df)), key="up_review_text")
    with c[1]:
        rating_col = pick("별점 열", {"rating", "별점", "score", "평점", "stars"}, "up_review_rating")
    with c[2]:
        date_col = pick("작성일 열", {"created_at", "date", "작성일", "날짜"}, "up_review_date")
    with c[3]:
        brand_col = pick("브랜드 열", {"brand", "브랜드", "store_name", "가게"}, "up_review_brand")
    with c[4]:
        domain = DOMAINS[st.selectbox("도메인", list(DOMAINS), key="up_review_domain",
                                      help="배달: 배달 지연·누락·온도 등 / 온라인 쇼핑·제품: 배송 지연·오배송·품질·사이즈 등")]
    if st.button("리뷰 분석", type="primary", key="up_review_run"):
        conn = upload.new_db()
        try:
            with st.spinner("분석 중"):
                outcome = upload.analyze_reviews(
                    conn, df, {"text": text_col, "rating": None if rating_col == none else rating_col,
                               "created_at": None if date_col == none else date_col, "brand": None if brand_col == none else brand_col},
                    domain, settings,
                )
                upload.estimate(conn, settings)
            st.session_state["up_review"] = (conn, outcome.notes, domain)
        except ValueError as exc:
            st.error(str(exc))
    if "up_review" in st.session_state:
        conn, notes, domain = st.session_state["up_review"]
        _show_notes(notes)
        st.subheader("분석 결과")
        st.caption(criteria.domain_reliability(domain))
        page_review(conn, k="upres_", domain=domain, show_brand_filter=len(data.brands(conn, "review")) > 1)
        units = pd.read_sql_query("SELECT u.session_id, u.text, u.valence, u.arousal, u.aspect, u.journey_stage, u.scores_json, u.cx_signals_json FROM units u", conn)
        st.download_button("절별 결과 CSV 내려받기", units.to_csv(index=False).encode("utf-8-sig"), "voclens_review_units.csv", "text/csv", key="up_review_dl")


def upload_call_tab():
    st.markdown("상담 전사본을 올리거나 붙여 넣으면 화자 역할, 대화 흐름(끼어들기·침묵·반복), 감정 변화, 회복 계기를 분석합니다.")
    st.markdown("- **텍스트**: 한 줄에 `화자: 발화`. 줄 앞에 `[00:12]` 시각을 쓰면 그대로 쓰고, 없으면 글자 수로 길이를 추정합니다. 화자 이름이 상담원·고객이면 역할을 그대로 쓰고, A·B 같은 이름은 규칙으로 판정합니다.\n"
                "- **JSON**: [전사본 형식](https://github.com/jhheo51-arch/VoC-Lens/blob/main/docs/call_transcript_format.md)\n"
                "- **WAV**: 전사본과 파일 이름이 같으면(예: `call_01.txt` + `call_01.wav`) 목소리 높낮이·크기로 흥분 정도를 더합니다. 16kHz 모노 PCM WAV 권장.")
    files = st.file_uploader("전사본(.txt, .json)과 음성(.wav)", type=["txt", "json", "wav"], accept_multiple_files=True, key="up_call_files")
    if st.button("합성 샘플 전사본 붙여 넣기", key="up_call_sample"):
        st.session_state["up_call_text"] = _sample_call_text()
    pasted = st.text_area("또는 전사본 붙여 넣기", height=180, key="up_call_text", placeholder="상담원: 네 고객센터입니다 무엇을 도와드릴까요\n고객: 한 시간 전에 주문했는데 아직도 안 와요")
    domain = DOMAINS[st.selectbox("도메인", list(DOMAINS), key="up_call_domain")]
    if st.button("상담 분석", type="primary", key="up_call_run"):
        calls, audio, errors = [], {}, []
        for f in files or []:
            name = f.name
            if name.lower().endswith(".wav"):
                audio[Path(name).stem] = f.getvalue()
                continue
            try:
                calls.append(upload.load_call_upload(name, f.getvalue()))
            except Exception as exc:  # 형식 오류는 파일별로 알려 주고 나머지는 계속 분석
                errors.append(f"{name}: {exc}")
        if pasted.strip():
            try:
                calls.append(upload.parse_text_transcript(pasted, call_id="pasted_call"))
            except ValueError as exc:
                errors.append(f"붙여 넣은 전사본: {exc}")
        audio_only = [n for n in audio if n not in {c.call_id for c in calls}]
        for n in audio_only:
            try:
                calls.append(upload.transcribe_audio_only(f"{n}.wav", audio.pop(n)))
            except RuntimeError:
                errors.append(f"{n}.wav: 전사본이 없어 음성 전사(STT)가 필요합니다. 이 배포본에는 STT 모델이 없습니다. 같은 이름의 전사본을 함께 올리거나, 로컬에서 `pip install -e .[audio]` 후 실행해 주세요.")
        for e in errors:
            st.error(e)
        if calls:
            conn = upload.new_db()
            with st.spinner("분석 중"):
                outcome = upload.analyze_calls(conn, calls, audio, settings, domain)
                upload.estimate(conn, settings)
            st.session_state["up_call"] = (conn, outcome.notes, domain)
    if "up_call" in st.session_state:
        conn, notes, domain = st.session_state["up_call"]
        _show_notes(notes)
        st.subheader("분석 결과")
        page_call(conn, k="upres_", domain=domain)


def upload_video_tab():
    st.markdown("영상 자막 파일(VTT·SRT)과 댓글을 올리면 내용 구간, 구간별 감정, 타임스탬프 댓글 반응을 분석합니다. 사람이 단 자막을 권장합니다 (자동 생성 자막은 정확도가 낮음).")
    source = st.radio("입력 방식", ["자막 파일", "영상 번들 JSON", "YouTube 주소"], horizontal=True, key="up_video_source")
    bundle = None
    meta_inputs = {}
    if source == "자막 파일":
        c1, c2 = st.columns(2)
        sub = c1.file_uploader("자막 (.vtt, .srt)", type=["vtt", "srt"], key="up_video_sub")
        com = c2.file_uploader("댓글 (선택: .csv의 text 열 또는 한 줄에 하나인 .txt)", type=["csv", "txt"], key="up_video_com")
        if st.button("합성 샘플 자막·댓글로 해 보기", key="up_video_sample"):
            st.session_state["up_video_sample_data"] = True
        c3, c4, c5, c6 = st.columns([3, 1, 1, 1])
        meta_inputs = {
            "title": c3.text_input("영상 제목", value="무선청소기 한 달 사용 솔직 리뷰 장단점" if st.session_state.get("up_video_sample_data") else "", key="up_video_title"),
            "domain": DOMAINS[c4.selectbox("도메인", list(DOMAINS), index=1, key="up_video_domain")],
            "views": c5.number_input("조회수 (선택)", min_value=0, value=18234 if st.session_state.get("up_video_sample_data") else 0, key="up_video_views"),
            "likes": c6.number_input("좋아요 (선택)", min_value=0, value=512 if st.session_state.get("up_video_sample_data") else 0, key="up_video_likes"),
        }
        if st.button("영상 분석", type="primary", key="up_video_run"):
            try:
                if sub is not None:
                    sub_bytes = sub.getvalue()
                    comments = upload.read_comments(com.getvalue(), com.name) if com is not None else []
                elif st.session_state.get("up_video_sample_data"):
                    sub_bytes = _sample_vtt()
                    comments = upload.read_comments(_sample_comments().encode("utf-8"), "comments.txt")
                else:
                    st.error("자막 파일을 올려 주세요.")
                    return
                meta = {k: v for k, v in meta_inputs.items() if v}
                bundle = upload.bundle_from_subtitles(sub_bytes, comments, meta)
            except ValueError as exc:
                st.error(str(exc))
    elif source == "영상 번들 JSON":
        js = st.file_uploader("번들 JSON", type=["json"], key="up_video_json")
        if st.button("영상 분석", type="primary", key="up_video_run_json") and js is not None:
            try:
                bundle = upload.bundle_from_json(js.getvalue())
            except Exception as exc:
                st.error(f"번들을 읽을 수 없습니다: {exc}")
    else:
        url = st.text_input("YouTube 주소", key="up_video_url", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("사람이 단 한국어 자막이 있는 공개 영상만 됩니다. yt-dlp가 설치된 환경에서 동작하며, 댓글은 YOUTUBE_API_KEY가 있을 때만 가져옵니다. 이 배포본에서 안 되면 자막 파일로 올려 주세요.")
        if st.button("영상 분석", type="primary", key="up_video_run_url") and url:
            try:
                with st.spinner("자막·메타데이터 받는 중"):
                    bundle = upload.bundle_from_url(url)
            except Exception as exc:
                st.error(f"YouTube에서 가져오지 못했습니다: {exc}")
    if bundle is not None:
        conn = upload.new_db()
        with st.spinner("분석 중"):
            outcome = upload.analyze_video(conn, bundle, settings)
            upload.estimate(conn, settings)
        st.session_state["up_video"] = (conn, outcome.notes)
    if "up_video" in st.session_state:
        conn, notes = st.session_state["up_video"]
        _show_notes(notes)
        st.subheader("분석 결과")
        page_video(conn, k="upres_")


def page_upload():
    st.header("직접 분석")
    st.caption(
        f"올린 파일은 이 브라우저 세션의 메모리에서만 분석하고 서버에 저장하지 않습니다. 새로고침하면 사라집니다. "
        f"공개 배포라 리뷰 {upload.LIMITS['review_rows']:,}건, 전사본 {upload.LIMITS['calls']}개, 댓글 {upload.LIMITS['comments']}개까지 분석합니다. "
        "개인정보(전화번호·이메일·긴 숫자)는 분석 전에 가리지만, 민감한 실제 고객 데이터는 올리지 마세요."
    )
    tab_review, tab_call, tab_video = st.tabs(["리뷰", "상담", "영상"])
    with tab_review:
        upload_review_tab()
    with tab_call:
        upload_call_tab()
    with tab_video:
        upload_video_tab()


# =========================== 분석 기준 ===========================
def page_criteria():
    st.header("분석 기준")
    st.markdown("모든 판단은 아래 규칙과 사전으로 합니다. 표는 실제 설정 파일(`configs/`)에서 바로 만들어서 분석 결과와 항상 같습니다. "
                "지금은 사전·규칙 기반 기준선이며, 라벨 데이터가 생기면 학습 모델로 바꾸고 정확도를 비교합니다.")
    tabs = st.tabs(["감정·지수", "리뷰", "상담", "영상", "CX 지표", "정확도"])
    with tabs[0]:
        st.subheader("감정: 7감정 → 감성(V)·각성(A)")
        st.markdown("\n".join(f"- {r}" for r in criteria.emotion_rules()))
        st.dataframe(criteria.emotion_table(), hide_index=True, width="stretch")
        st.subheader("지수 6개")
        st.dataframe(criteria.INDEX_RULES, hide_index=True, width="stretch")
        st.caption("세션 지수 = 단위 점수 평균 × 100. 최고점, 임계치(0.5) 초과 비율, 마지막 값, 채널 내 백분위를 함께 저장합니다.")
    with tabs[1]:
        st.subheader("리뷰 분석 규칙")
        st.markdown("\n".join(f"- {r}" for r in criteria.review_rules()))
        st.subheader("불만 유형 분류 기준")
        st.markdown("\n".join(f"- {r}" for r in criteria.taxonomy_rules()))
        st.dataframe(criteria.taxonomy_table(), hide_index=True, width="stretch")
        st.subheader("CX 신호")
        st.dataframe(criteria.signal_table(), hide_index=True, width="stretch")
    with tabs[2]:
        st.subheader("상담 분석 규칙")
        st.markdown("\n".join(f"- {r}" for r in criteria.call_rules()))
        t = criteria.call_tables()
        st.subheader("상담 표현 사전")
        st.dataframe(t["phrases"], hide_index=True, width="stretch")
        st.subheader("음성 각성 가중치")
        st.dataframe(t["arousal_weights"], hide_index=True)
    with tabs[3]:
        st.subheader("영상 분석 규칙")
        st.markdown("\n".join(f"- {r}" for r in criteria.video_rules()))
        t = criteria.video_tables()
        left, right = st.columns(2)
        with left:
            st.markdown("**영상 유형 키워드**")
            st.dataframe(t["types"], hide_index=True, width="stretch")
        with right:
            st.markdown("**구간 이름 키워드**")
            st.dataframe(t["segment_labels"], hide_index=True, width="stretch")
    with tabs[4]:
        st.subheader("CX 지표 추정과 채널 비교")
        st.markdown("\n".join(f"- {r}" for r in criteria.cx_rules()))
    with tabs[5]:
        st.subheader("실제 데이터로 잰 정확도")
        st.markdown("공개 실제 데이터(네이버 쇼핑 리뷰 20만 건, KOTE 댓글 5만 문장)로 기준선이 어디까지 맞는지 쟀습니다. "
                    "[평가 문서](https://github.com/jhheo51-arch/VoC-Lens/blob/main/docs/evaluation.md)")
        # 작은 표라 st.table로 그린다 (숨겨진 탭 안의 st.dataframe은 처음에 폭이 0으로 잡히는 경우가 있다)
        st.table(criteria.accuracy_summary().set_index("평가"))
        st.subheader("불만 유형별 (온라인 쇼핑, 분류 기준 v0.3.0)")
        by_type = criteria.accuracy_by_type().set_index("불만 유형")
        st.table(by_type.style.format({c: lambda v: "—" if pd.isna(v) else f"{v:.2f}" for c in ("정밀도", "재현율", "F1")}))
        st.markdown("\n".join(f"- {r}" for r in criteria.accuracy_notes()))


if page == "홈":
    conn = sample_conn()
    if not data.counts(conn):
        st.info("분석된 데이터가 없습니다. `voclens review / call / video` 명령으로 데이터를 넣거나, VOCLENS_DASHBOARD_ADMIN=1로 실행해 사이드바의 **합성 샘플로 다시 분석**을 누르세요.")
        st.stop()
    st.header("예시 결과")
    st.caption("합성 데이터(리뷰 180건, 상담 4건, 영상 2건)로 만든 결과입니다. **직접 분석**에서 내 리뷰·상담·영상을 올리면 리뷰·상담·영상 탭과 같은 화면으로 결과가 나옵니다.")
    tabs = st.tabs(HOME_TABS)
    for tab, render in zip(tabs, (page_home, page_review, page_call, page_video, page_compare)):
        with tab:
            render(conn)
    conn.close()
elif page == "직접 분석":
    page_upload()
else:
    page_criteria()
