"""대시보드 차트 (Altair).

색 규칙 (dataviz 기준 팔레트)
  범주: 고정 순서 슬롯을 개체에 묶는다. 채널은 리뷰=1(파랑), 상담=2(주황), 영상=3(청록),
        역할은 고객=1, 상담원=2. 처음 세 슬롯은 모든 쌍 조합에서 색약 구분 검증을 통과한다.
  극성(Valence): 파랑(긍정) ↔ 빨강(부정). 막대는 부호로 색을 고르고 길이로 크기를 보인다.
  크기(비중): 파랑 한 가지 색의 밝기 단계.
모든 차트는 축 하나, 마우스를 올리면 값이 보이는 툴팁을 단다.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POSITIVE, NEGATIVE, NEUTRAL = "#2a78d6", "#e34948", "#c3c2b7"
# 라벨 글자는 밝은·어두운 테마 모두에서 읽히는 보조 잉크를 쓴다 (계열 색으로 칠하지 않는다)
MUTED_INK = "#898781"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CHANNEL_COLORS = {"리뷰": SERIES[0], "상담": SERIES[1], "영상(크리에이터)": SERIES[2], "영상 댓글": SERIES[3]}
ROLE_COLORS = {"고객": SERIES[0], "상담원": SERIES[1], "크리에이터": SERIES[0], "게스트": SERIES[1], "댓글": SERIES[2]}


def _sign_color(field: str) -> alt.Color:
    return alt.condition(f"datum['{field}'] >= 0", alt.value(POSITIVE), alt.value(NEGATIVE))


def diverging_bars(df: pd.DataFrame, category: str, value: str, title: str, n_field: str | None = None) -> alt.Chart:
    """범주별 평균 Valence(-1~1). 0 기준선, 부호로 색, 길이로 크기."""
    tooltip = [alt.Tooltip(category, title="항목"), alt.Tooltip(value, title="평균 감성", format="+.2f")]
    if n_field:
        tooltip.append(alt.Tooltip(n_field, title="절 수"))
    bars = (
        alt.Chart(df, title=title)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.6})
        .encode(
            x=alt.X(value, title="평균 감성 (-1 부정 ~ +1 긍정)", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y(category, title=None, sort=alt.EncodingSortField(value, order="ascending")),
            color=_sign_color(value),
            tooltip=tooltip,
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=NEUTRAL).encode(x="x")
    # 전체 높이 대신 범주당 높이(step)를 정해야 축 라벨이 겹쳐 숨지 않는다
    return (bars + zero).properties(height=alt.Step(30))


def priority_quadrant(pains: pd.DataFrame) -> alt.Chart:
    """개선 과제 4분면: 가로 = 실행 난이도, 세로 = 우선순위, 점 크기 = 언급 비율."""
    df = pains.assign(share_pct=pains["share"] * 100)
    # 같은 난이도의 점이 세로로 겹치지 않게 우선순위 순서대로 좌우로 조금씩 벌린다
    rank = df.groupby("difficulty")["priority"].rank(ascending=False, method="first") - 1
    size = df.groupby("difficulty")["priority"].transform("size")
    df["x"] = df["difficulty"] + (rank - (size - 1) / 2) * 0.18
    median = float(df["priority"].median())
    base = alt.Chart(df, title="개선 과제 우선순위 × 실행 난이도")
    points = base.mark_circle(color=SERIES[0], opacity=0.85, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("x:Q", title="실행 난이도 (1 쉬움 ~ 3 어려움)", scale=alt.Scale(domain=[0.5, 3.6]), axis=alt.Axis(values=[1, 2, 3], format="d")),
        y=alt.Y("priority:Q", title="우선순위 (0~100)", scale=alt.Scale(domain=[0, 105])),
        size=alt.Size("share_pct:Q", title="언급 비율 (%)", scale=alt.Scale(range=[80, 900]), legend=alt.Legend(orient="top", values=[5, 10, 15])),
        tooltip=[
            alt.Tooltip("label", title="불만 유형"),
            alt.Tooltip("quadrant", title="4분면"),
            alt.Tooltip("priority", title="우선순위", format=".1f"),
            alt.Tooltip("share_pct", title="언급 비율 (%)", format=".1f"),
            alt.Tooltip("difficulty", title="난이도"),
        ],
    )
    labels = base.mark_text(align="left", dx=14, fontSize=12, color=MUTED_INK).encode(x="x:Q", y="priority:Q", text="label")
    rule = alt.Chart(pd.DataFrame({"y": [median]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y="y")
    return (rule + points + labels).properties(height=420)


def grouped_share(df: pd.DataFrame, category: str, group: str, value: str, title: str) -> alt.Chart:
    """그룹(브랜드)별 비중 비교 막대. 그룹 색은 정렬된 이름 순서로 슬롯에 고정한다."""
    groups = sorted(df[group].dropna().unique())
    return (
        alt.Chart(df, title=title)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(value, title="언급 비율 (%)"),
            y=alt.Y(category, title=None, sort="-x"),
            yOffset=alt.YOffset(group, sort=groups),
            color=alt.Color(group, title=None, scale=alt.Scale(domain=groups, range=SERIES[: len(groups)]), legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip(group, title="그룹"), alt.Tooltip(category, title="항목"), alt.Tooltip(value, title="비율 (%)", format=".1f")],
        )
        .properties(height=alt.Step(18))
    )


def emotion_timeline(units: pd.DataFrame, events: pd.DataFrame | None = None, roles=("고객", "상담원")) -> alt.Chart:
    """발화별 Valence 시간축 (상담: 고객·상담원). 회색 띠 = 감정 급변 구간."""
    df = units[units["role"].isin(roles)].copy()
    df["text_short"] = df["text"].str.slice(0, 60)
    domain = [r for r in roles if r in set(df["role"])]
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True, stroke="#fcfcfb", strokeWidth=2))
        .encode(
            x=alt.X("start_sec:Q", title="통화 시간 (초)"),
            y=alt.Y("valence:Q", title="감성 (-1 부정 ~ +1 긍정)", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("role:N", title=None, scale=alt.Scale(domain=domain, range=[ROLE_COLORS[r] for r in domain]), legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip("role", title="화자"),
                alt.Tooltip("start_sec", title="시작 (초)", format=".1f"),
                alt.Tooltip("valence", title="감성", format="+.2f"),
                alt.Tooltip("text_short", title="발화"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=NEUTRAL).encode(y="y")
    layers = [zero, line]
    if events is not None and not events.empty:
        ev = events.assign(direction=events["delta"].map(lambda d: "상승" if d > 0 else "하락"))
        band = (
            alt.Chart(ev)
            .mark_rect(color=NEUTRAL, opacity=0.25)
            .encode(
                x="start_sec:Q",
                x2="end_sec:Q",
                tooltip=[
                    alt.Tooltip("index_name", title="지표"),
                    alt.Tooltip("direction", title="방향"),
                    alt.Tooltip("delta", title="변화량", format="+.2f"),
                    alt.Tooltip("trigger_text", title="직전 상담원 발화"),
                ],
            )
        )
        layers.insert(0, band)
    return alt.layer(*layers).properties(height=320)


def video_timeline(units: pd.DataFrame, segments: pd.DataFrame) -> alt.Chart:
    """영상 문장 감성(파랑 선) + 구간 띠 + 타임스탬프 댓글(청록 점)."""
    speech = units[units["speaker_role"].isin(["creator", "guest"])].assign(text_short=lambda d: d["text"].str.slice(0, 60))
    comments = units[(units["speaker_role"] == "commenter") & units["start_sec"].notna()].assign(
        text_short=lambda d: d["text"].str.slice(0, 60)
    )
    layers = []
    if not segments.empty:
        seg = segments.assign(shade=[i % 2 for i in range(len(segments))])
        layers.append(
            alt.Chart(seg)
            .mark_rect(opacity=0.12)
            .encode(
                x="start_sec:Q",
                x2="end_sec:Q",
                color=alt.Color("shade:N", scale=alt.Scale(domain=[0, 1], range=[NEUTRAL, "#ffffff"]), legend=None),
                tooltip=[alt.Tooltip("name", title="구간"), alt.Tooltip("valence", title="구간 평균 감성", format="+.2f")],
            )
        )
        layers.append(
            alt.Chart(seg).mark_text(baseline="top", align="left", dx=4, dy=4, fontSize=11, color=MUTED_INK).encode(
                x=alt.X("start_sec:Q"), y=alt.value(4), text="label"
            )
        )
    layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=NEUTRAL).encode(y="y"))
    speech = speech.assign(role=speech["speaker_role"].map({"creator": "크리에이터", "guest": "게스트"}))
    roles = [r for r in ("크리에이터", "게스트") if r in set(speech["role"])]
    # 선은 크리에이터만 잇고(발화 흐름), 게스트 발화는 점으로만 표시한다
    layers.append(
        alt.Chart(speech[speech["role"] == "크리에이터"])
        .mark_line(strokeWidth=2, color=ROLE_COLORS["크리에이터"])
        .encode(x=alt.X("start_sec:Q", title="영상 시간 (초)"), y=alt.Y("valence:Q", title="감성 (-1 부정 ~ +1 긍정)", scale=alt.Scale(domain=[-1, 1])))
    )
    layers.append(
        alt.Chart(speech)
        .mark_circle(size=70, opacity=1, stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x="start_sec:Q",
            y="valence:Q",
            color=alt.Color("role:N", title=None, scale=alt.Scale(domain=roles, range=[ROLE_COLORS[r] for r in roles]), legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("role", title="화자"), alt.Tooltip("start_sec", title="시작 (초)", format=".0f"), alt.Tooltip("valence", title="감성", format="+.2f"), alt.Tooltip("text_short", title="문장")],
        )
    )
    if not comments.empty:
        layers.append(
            alt.Chart(comments)
            .mark_point(shape="diamond", size=180, filled=True, opacity=1, color=SERIES[2], stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x="start_sec:Q",
                y="valence:Q",
                tooltip=[alt.Tooltip("start_sec", title="댓글이 가리킨 시각 (초)", format=".0f"), alt.Tooltip("valence", title="댓글 감성", format="+.2f"), alt.Tooltip("text_short", title="댓글")],
            )
        )
    # 구간 띠의 색 척도와 화자 색 척도가 합쳐지지 않게 층마다 따로 둔다
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=340)


def share_heatmap(mix: pd.DataFrame, labels: dict[str, str]) -> alt.Chart:
    """채널 × 불만 유형 구성비 (한 색 밝기 단계). 채널 순서는 labels의 순서를 따른다."""
    long = mix.reset_index().melt(id_vars="channel", var_name="불만 유형", value_name="share")
    long["채널"] = long["channel"].map(labels).fillna(long["channel"])
    order = list(mix.sum(axis=0).sort_values(ascending=False).index)
    return (
        alt.Chart(long, title="채널별 불만 유형 구성비 (%)")
        .mark_rect(stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("채널:N", title=None, sort=[v for k, v in labels.items() if k in mix.index], axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("불만 유형:N", title=None, sort=order),
            color=alt.Color("share:Q", title="비중 (%)", scale=alt.Scale(range=SEQUENTIAL)),
            tooltip=[alt.Tooltip("채널"), alt.Tooltip("불만 유형"), alt.Tooltip("share", title="비중 (%)", format=".1f")],
        )
        .properties(height=alt.Step(28))
    )


def lift_bars(df: pd.DataFrame) -> alt.Chart:
    """회복 직전 상담원 표현의 lift. 1 = 전체 상담원 발화와 같은 비율."""
    bars = (
        alt.Chart(df, title="회복 직전 상담원 표현 (lift)")
        .mark_bar(cornerRadiusEnd=4, color=SERIES[1])
        .encode(
            x=alt.X("lift:Q", title="lift (1보다 크면 회복 직전에 더 자주 나옴)"),
            y=alt.Y("category:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("category", title="표현 유형"),
                alt.Tooltip("lift", title="lift", format=".2f"),
                alt.Tooltip("n_trigger", title="회복 직전 건수"),
                alt.Tooltip("n_agent_turns", title="전체 상담원 발화 수"),
            ],
        )
    )
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(x="x")
    return (bars + one).properties(height=alt.Step(32))
