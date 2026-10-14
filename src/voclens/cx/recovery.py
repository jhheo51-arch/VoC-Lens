"""회복 계기 분석: 고객 감정이 크게 오르기 직전 상담원이 한 말에는 어떤 표현이 많은가.

상담원 발화를 표현 유형(공감·사과 / 해결 조치 / 이관 / 보류)으로 나누고,
회복 직전 발화에서의 비율을 전체 상담원 발화에서의 비율과 비교한다 (lift = 회복 직전 비율 / 전체 비율).
lift가 1보다 크면 그 표현 뒤에 회복이 자주 일어난다는 뜻이다. 인과가 아니라 연관이며,
표본이 적으면 우연일 수 있어서 건수를 함께 본다.
"""

from __future__ import annotations

import re
import sqlite3

from voclens.config import load_yaml

CATEGORIES = {"empathy": "공감·사과", "resolution": "해결 조치", "transfer": "이관", "pending": "보류"}


def _patterns() -> dict[str, re.Pattern]:
    data = load_yaml("lexicons/call_phrases.yaml")
    return {k: re.compile("|".join(f"(?:{p})" for p in data[k])) for k in CATEGORIES}


def categorize(text: str, patterns: dict[str, re.Pattern] | None = None) -> list[str]:
    patterns = patterns or _patterns()
    found = [k for k, p in patterns.items() if p.search(text)]
    return found or ["other"]


def recovery_trigger_summary(conn: sqlite3.Connection, min_delta: float = 0.0) -> list[dict]:
    patterns = _patterns()
    agent = conn.execute("SELECT unit_id, text FROM units WHERE source_type = 'call' AND speaker_role = 'agent'").fetchall()
    triggers = conn.execute(
        "SELECT DISTINCT e.trigger_unit_id FROM events e"
        " WHERE e.index_name = 'valence' AND e.delta > ? AND e.trigger_unit_id IS NOT NULL",
        (min_delta,),
    ).fetchall()
    trigger_ids = {r[0] for r in triggers}
    if not agent:
        return []

    labels = {r["unit_id"]: categorize(r["text"], patterns) for r in agent}
    rows = []
    for key in [*CATEGORIES, "other"]:
        n_all = sum(key in v for v in labels.values())
        n_trigger = sum(key in labels.get(uid, []) for uid in trigger_ids)
        share_all = n_all / len(labels)
        share_trigger = n_trigger / len(trigger_ids) if trigger_ids else 0.0
        rows.append(
            {
                "category": CATEGORIES.get(key, "기타"),
                "n_trigger": n_trigger,
                "n_agent_turns": n_all,
                "share_trigger": round(share_trigger, 3),
                "share_all": round(share_all, 3),
                "lift": round(share_trigger / share_all, 2) if share_all else None,
            }
        )
    return sorted(rows, key=lambda r: -(r["lift"] or 0))
