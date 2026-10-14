"""상담 통화 단위 지수.

리뷰와 같은 6개 지수를 **고객 발화만으로** 계산한 뒤, 좌절 지수에 대화 역학을 더한다.
  좌절 += 끼어들기(최대 10점) + 같은 말 반복(최대 15점) + 최장 침묵(최대 10점, 30초에서 최대)
감정 급변 이벤트는 고객 Valence와 스트레스 시계열에서 찾고, 직전 상담원 발화를 계기로 붙인다.
"""

from __future__ import annotations

from voclens.audio.dynamics import DynamicsResult
from voclens.indices.compute import score_session
from voclens.indices.events import detect_changes
from voclens.schema import Event, Session, Unit


def _frustration_boost(d: DynamicsResult) -> float:
    return (
        10 * min(d.customer_interruptions, 3) / 3
        + 15 * min(d.repeat_count, 3) / 3
        + 10 * min(d.max_silence / 30, 1)
    )


def _trigger(units: list[Unit], before_order: int) -> str | None:
    agent = [u for u in units if u.speaker_role == "agent" and u.order < before_order]
    return agent[-1].unit_id if agent else None


def call_events(session: Session, call_settings: dict) -> list[Event]:
    customer = [u for u in session.units if u.speaker_role == "customer"]
    events: list[Event] = []
    series = {
        "valence": [u.valence or 0.0 for u in customer],
        "stress": [u.scores.get("stress", 0.0) for u in customer],
    }
    for name, values in series.items():
        for k, change in enumerate(
            detect_changes(
                values,
                window=call_settings["event_window"],
                min_delta=call_settings["event_min_delta"],
                alpha=call_settings["event_smoothing"],
            )
        ):
            first, last = customer[change.start], customer[change.end]
            # 계기: 실제로 가장 크게 변한 한 걸음(고객 발화 i-1 → i) 직전의 상담원 발화
            sign = 1 if change.delta > 0 else -1
            pivot = max(
                range(change.start + 1, change.end + 1),
                key=lambda i: sign * (values[i] - values[i - 1]),
                default=change.end,
            )
            events.append(
                Event(
                    event_id=f"{session.session_id}!{name}{k}",
                    index_name=name,
                    start_sec=first.start_sec,
                    end_sec=last.end_sec,
                    delta=change.delta,
                    start_unit_id=first.unit_id,
                    end_unit_id=last.unit_id,
                    trigger_unit_id=_trigger(session.units, customer[pivot].order),
                )
            )
    return events


def score_call(session: Session, dynamics: DynamicsResult, settings: dict) -> Session:
    idx = settings["indices"]
    customer = [u for u in session.units if u.speaker_role == "customer"]
    score_session(session, idx["above_threshold"], idx["recovery_min_negative"], units=customer)

    frustration = session.indices.get("frustration")
    if frustration and frustration.raw is not None:
        frustration.raw = round(min(100.0, frustration.raw + _frustration_boost(dynamics)), 2)

    session.events = call_events(session, settings["call"]) if customer else []
    return session
