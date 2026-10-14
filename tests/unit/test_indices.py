import pytest

from voicecx.indices.compute import (
    assign_percentiles,
    engagement_index,
    recovery_index,
    score_session,
    unit_scores,
)
from voicecx.schema import Session, SourceType, Unit


def u(order, valence, arousal=0.3, text="텍스트", signals=None, aspect=None):
    return Unit(
        unit_id=f"u{order}", session_id="s", source_type=SourceType.REVIEW, order=order,
        text=text, valence=valence, arousal=arousal, cx_signals=signals or {}, aspect=aspect,
    )


def test_anger_is_stress_sadness_is_negative_affect():
    anger = unit_scores(u(0, -0.7, 0.85))
    sadness = unit_scores(u(0, -0.7, 0.25))
    assert anger["stress"] > anger["negative_affect"]
    assert sadness["negative_affect"] > sadness["stress"]


def test_positive_unit_has_no_negative_scores():
    s = unit_scores(u(0, 0.6, 0.6))
    assert s["happiness"] == pytest.approx(0.75)
    assert s["stress"] == s["negative_affect"] == 0


def test_frustration_uses_signals():
    plain = unit_scores(u(0, -0.3))
    effort = unit_scores(u(0, -0.3, signals={"repeat_contact": True, "waiting": True}))
    assert effort["frustration"] > plain["frustration"]
    assert 0 <= effort["frustration"] <= 1


def test_recovery_none_without_initial_negative():
    assert recovery_index([u(0, 0.3), u(1, 0.5)]) is None


def test_recovery_detects_improvement_and_worsening():
    assert recovery_index([u(0, -0.6), u(1, 0.6)]) > 90
    assert recovery_index([u(0, -0.3), u(1, -0.3)]) == 50
    assert recovery_index([u(0, -0.3), u(1, 0.0), u(2, -0.8)]) < 50


def test_engagement_rewards_detail_and_intent():
    short = engagement_index([u(0, 0.5, text="굿")])
    detailed = engagement_index([
        u(0, 0.5, text="치킨이 바삭하고 양도 많았어요 " * 3, aspect="맛"),
        u(1, 0.5, text="배달도 빨랐어요", aspect="배달", signals={"repurchase_intent": True}),
    ])
    assert detailed > short


def test_score_session_fills_all_indices_and_peak_end():
    session = Session(session_id="s", source_id="r1", source_type=SourceType.REVIEW,
                      units=[u(0, -0.7, 0.85), u(1, 0.6, 0.6)])
    score_session(session)
    assert set(session.indices) == {"stress", "happiness", "negative_affect", "frustration", "recovery", "engagement"}
    assert session.indices["stress"].peak > session.indices["stress"].end_value
    assert session.peak_end is not None


def test_score_session_empty():
    session = score_session(Session(session_id="s", source_id="r", source_type=SourceType.REVIEW))
    assert session.indices["stress"].raw is None


def test_assign_percentiles_within_channel():
    sessions = []
    for i, v in enumerate([-0.8, -0.2, 0.5]):
        s = Session(session_id=f"s{i}", source_id=f"r{i}", source_type=SourceType.REVIEW, units=[u(0, v, 0.8)])
        sessions.append(score_session(s))
    assign_percentiles(sessions)
    stress = [s.indices["stress"].percentile for s in sessions]
    assert stress[0] > stress[1] >= stress[2]
