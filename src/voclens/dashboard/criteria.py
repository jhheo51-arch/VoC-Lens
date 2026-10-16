"""분석 기준 화면용 표: 설정 파일(configs/)에서 바로 만들어서 실제 규칙과 항상 같다."""

from __future__ import annotations

import pandas as pd

from voclens.audio.ser import AROUSAL_WEIGHTS, BASELINE_AROUSAL
from voclens.config import load_settings, load_yaml
from voclens.cx.compare import SMALL_CHANNEL_MENTIONS
from voclens.text.cx_signals import SIGNAL_LABELS

INDEX_RULES = pd.DataFrame(
    [
        ("스트레스", "부정 × 높은 각성", "max(0, −V) × A ÷ 0.6", "짜증·분노가 클수록 높음"),
        ("행복", "긍정", "max(0, V) ÷ 0.8", "만족·기쁨 표현이 많을수록 높음"),
        ("부정 정서", "부정 × 낮은 각성", "max(0, −V) × (1 − A) ÷ 0.5", "실망·체념. 의료 진단이 아님"),
        ("좌절", "고객 노력", "0.40·반복 문의 + 0.25·대기 + 0.15·이탈/비추천 + 0.40·max(0, −V)", "상담은 끼어들기(최대 10점)·반복(15점)·최장 침묵(10점) 가산"),
        ("감정 회복", "후반부 − 전반부 감성", "50 + 50 × (후반 평균 V − 전반 평균 V)", "전반부 최저 V가 −0.2보다 낮을 때만 계산. 50 = 변화 없음"),
        ("몰입", "관심과 열의", "0.35·글 길이(150자=최대) + 0.35·속성 수(3개=최대) + 0.30·재주문/추천 의사", "영상은 0.5·좋아요율(5%=최대) + 0.3·조회 1천 회당 댓글(5개=최대) + 0.2·타임스탬프 댓글 비율"),
    ],
    columns=["지수 (0~100)", "뜻", "단위 점수 공식 (V = 감성 −1~1, A = 각성 0~1)", "비고"],
)


def emotion_table() -> pd.DataFrame:
    lex = load_yaml("lexicons/emotion.yaml")
    rows = []
    for emotion, (v, a) in lex["circumplex"].items():
        entries = lex["emotions"].get(emotion, [])
        examples = ", ".join(p.replace("\\", "") for p, _ in entries[:8])
        rows.append({"감정": emotion, "감성 V": v, "각성 A": a, "사전 표현 수": len(entries), "예시 표현": examples or "(감정 표현이 없을 때 기본 점수 1.0)"})
    return pd.DataFrame(rows)


def emotion_rules() -> list[str]:
    lex = load_yaml("lexicons/emotion.yaml")
    return [
        "절마다 사전 표현이 걸린 감정에 가중치를 더하고, 중립에 기본 점수 1.0을 더한 뒤 합이 1이 되게 나눠 7감정 확률을 만든다.",
        "감성(V)·각성(A)은 7감정 확률로 위 표의 좌표를 가중평균한 값이다.",
        f"강조 표현({', '.join(lex['intensifiers'])})이 바로 앞에 오면 가중치 1.5배.",
        f"부정 표현(\"안 좋아요\", \"좋지 않\", \"추천 안 함\")이 붙으면 긍정은 {lex['flip_positive_to'][0]}(×{lex['flip_positive_to'][1]})로, 부정 감정은 {lex['flip_negative_to'][0]}(×{lex['flip_negative_to'][1]})로 바꿔 센다.",
        "가정·기대 미충족(\"빨랐으면 재주문할 텐데\", \"더 좋아야\")은 긍정 표현에만 적용해 부정으로 바꿔 센다.",
        "명시적 불만 표현이 있는 절은 감정 사전이 놓쳐도 감성을 −0.3 이하로 본다 (예: \"1시간 만에 왔네요\").",
    ]


def taxonomy_table() -> pd.DataFrame:
    tx = load_yaml("taxonomy.yaml")
    domain_names = {"delivery": "배달", "product": "제품"}
    rows = []
    for name, sub in tx["subcategories"].items():
        rows.append(
            {
                "대분류": sub["category"],
                "소분류": name,
                "도메인": ", ".join(domain_names.get(d, d) for d in sub.get("domains", [])) or "전체",
                "불만 표현 (감정과 상관없이 불만)": ", ".join(p.replace("\\s?", " ").replace("\\s*", " ").replace("\\d+", "N") for p in sub.get("complaint", [])[:6]),
                "주제 표현 (부정 감성일 때만)": ", ".join(p.replace("\\s?", " ") for p in sub.get("topic", [])[:5]),
                "난이도": sub.get("difficulty"),
                "여정 단계": sub.get("stage") or "—",
            }
        )
    return pd.DataFrame(rows)


def taxonomy_rules() -> list[str]:
    tx = load_yaml("taxonomy.yaml")
    return [
        f"분류 기준 버전 {tx.get('version')}. 한 절에 여러 소분류가 붙을 수 있다 (다중 라벨).",
        "데이터의 도메인(배달·제품)을 알면 그 도메인 소분류와 \"전체\" 소분류만 쓴다.",
        f"불만이 아닌 절: 감성 {tx['praise_min_valence']} 이상은 \"{tx['praise_label']}\", {tx['unknown_max_valence']} 이하인데 맞는 소분류가 없으면 \"{tx['unknown_label']}\"(신규 이슈 후보).",
        "난이도: 1 = 매장 절차로 바로 고칠 수 있음, 2 = 레시피·인력·정책 조정, 3 = 외부 요인·구조적 투자.",
    ]


def signal_table() -> pd.DataFrame:
    data = load_yaml("lexicons/cx_signals.yaml")
    return pd.DataFrame(
        [
            {"신호": SIGNAL_LABELS.get(k, k), "표현 예시": ", ".join(v["patterns"][:6]), "제외 표현": ", ".join(v.get("exclude", [])[:4]) or "—"}
            for k, v in data["signals"].items()
        ]
    )


def review_rules() -> list[str]:
    report = load_settings()["report"]
    return [
        "정제: 반복 문자 정규화(ㅋㅋㅋㅋ → ㅋㅋ), 이모티콘·이모지 → 감정 토큰([슬픔] 등), 전화번호·이메일·8자리 이상 숫자 마스킹.",
        "리뷰 이벤트(\"리뷰이벤트\", \"서비스 감사합니다\" 등)가 있는 리뷰는 표시하고, 그 문구 절은 지수 계산에서 뺀다.",
        "절 분리: 종결어미(요·다·음·임·함·됨·네·죠) 어절과 문장부호에서 문장을, \"~는데\", \"~지만\", \"근데\", \"하지만\"에서 절을 나눈다.",
        "우선순위 = 언급 비율(최대값 기준 정규화) × (0.5 + 감정 강도) × (1 + 0.5 × 증가 추세) × (1 + 이탈 의사 초과분 + 재주문 의사 부족분).",
        "증가 추세: 작성일이 70% 이상 있을 때, 기간 마지막 1/3의 언급 비율을 앞 2/3와 비교한다.",
        "4분면: 우선순위가 중앙값 이상이면 영향 큼, 난이도 1이면 실행 쉬움 → Quick Win / 핵심 과제 / 소규모 개선 / 모니터링.",
        f"브랜드 비교: 지수는 베이지안 축소(가상 표본 {report['shrinkage_k']}개)로 보정, 신뢰구간은 부트스트랩 {report['bootstrap_iterations']}회, 차이는 카이제곱(Cramér's V)·Mann-Whitney U.",
        "별점–감정 불일치: 리뷰 평균 감성으로 별점을 예측하는 회귀의 표준화 잔차가 ±1.5 이상이면서 방향이 반대인 리뷰.",
    ]


def call_tables() -> dict[str, pd.DataFrame]:
    phrases = load_yaml("lexicons/call_phrases.yaml")
    names = {"agent_script": "상담원 정형 표현 (역할 판정 +1)", "customer_cue": "고객 표현 (역할 판정 −0.5)", "empathy": "공감·사과", "resolution": "해결 조치 → 결과 '해결'", "transfer": "이관 → 결과 '이관'", "pending": "보류 → 결과 '보류'"}
    phrase_df = pd.DataFrame(
        [{"구분": names.get(k, k), "표현 예시": ", ".join(p.replace("\\s?", " ").replace("\\s", " ") for p in v[:6])} for k, v in phrases.items()]
    )
    weights = pd.DataFrame(
        [{"음향 특징 (화자 기준 z-score)": k, "각성 가중치": w} for k, w in AROUSAL_WEIGHTS.items()]
    )
    return {"phrases": phrase_df, "arousal_weights": weights}


def call_rules() -> list[str]:
    c = load_settings()["call"]
    return [
        "역할 판정: 화자마다 (상담원 정형 표현 수 − 0.5 × 고객 표현 수) ÷ 발화 수, 첫 발화자에 +0.3. 가장 높은 화자가 상담원.",
        f"긴 침묵: 앞 발화가 끝나고 {c['long_silence_sec']}초 이상 무음. 고객 발화 직전 5초 이상이면 '대기' 신호.",
        f"끼어들기: 상대 발화가 끝나기 {c['interruption_overlap_sec']}초 전보다 먼저 시작.",
        f"같은 말 반복: 고객 발화끼리 글자 2-gram 겹침 계수(|A∩B| ÷ min(|A|,|B|))가 {c['repeat_similarity']} 이상 ({c['repeat_min_chars']}자 이상 발화만).",
        "통화 결과: 상담원 발화 뒤쪽 40%에서 마지막으로 나온 조치(이관 > 보류 > 해결 순으로 확인).",
        "지수는 고객 발화만으로 계산한다. Peak-End = (가장 나빴던 순간 + 마지막 순간의 부정 경험) ÷ 2.",
        f"감정 급변: 고객 감성 시계열을 지수이동평균(α={c['event_smoothing']})으로 평활화하고 {c['event_window']}칸 전 대비 {c['event_min_delta']} 이상 변한 구간. 가장 크게 변한 한 걸음 직전 상담원 발화를 계기로 기록.",
        f"음성 융합: 각성은 음향 {c['fusion']['arousal'][0]} / 텍스트 {c['fusion']['arousal'][1]}, 감성은 음향 {c['fusion']['valence'][0]} / 텍스트 {c['fusion']['valence'][1]}. 각 가중치에 모달 신뢰도(STT 신뢰도, SNR, 발화 길이 1초 이상)를 곱한다.",
        f"음성 각성 기준선: 화자의 처음 {c['acoustic_baseline_sec']}초 발화로 평소 목소리를 정하고, 평소 각성을 {BASELINE_AROUSAL}로 둔 로지스틱 함수에 아래 가중치를 넣는다. 음성은 감성(V)을 내지 않는다.",
    ]


def video_tables() -> dict[str, pd.DataFrame]:
    v = load_yaml("video.yaml")
    types = pd.DataFrame([{"유형": spec["label"], "키워드": ", ".join(spec["keywords"])} for spec in v["types"].values()])
    labels = pd.DataFrame([{"구간 이름": k, "키워드": ", ".join(kws)} for k, kws in v["segment_labels"].items()])
    return {"types": types, "segment_labels": labels}


def video_rules() -> list[str]:
    v = load_yaml("video.yaml")
    s = v["segmentation"]
    return [
        "자막: 사람이 단 한국어 자막만 쓴다(자동 자막 제외). 자막 큐를 이어 붙여 문장으로 나누고 글자 위치로 시각을 되찾는다.",
        f"상투 문구({', '.join(v['boilerplate'][:5])} …)가 있는 문장은 분석에서 뺀다.",
        "주 화자: 발화 시간이 가장 긴 화자가 크리에이터, 나머지는 게스트. 지수는 크리에이터·게스트 문장으로 계산.",
        "영상 유형: 제목·태그 키워드 ×2, 설명·자막 앞 10문장 ×1로 점수를 내 가장 높은 유형. 신뢰도 = (1위 − 2위) ÷ 1위.",
        f"구간 분할(TextTiling): 앞뒤 {s['block_size']}문장의 글자 2-gram 코사인 유사도가 국소 봉우리보다 {s['depth_threshold']} 이상 패인 곳과 전환 표현({', '.join(s['cue_phrases'][:6])} …) 앞을 경계로, 구간은 최소 {s['min_sentences']}문장.",
        "구간 이름: 구간 안 키워드가 가장 많은 이름. 마지막 구간에 총평 키워드가 있으면 총평, 첫 구간에 도입 키워드가 있으면 도입.",
        f"총평 가중 감성: 총평 구간 문장에 가중치 {v['verdict_weight']}.",
        f"타임스탬프 댓글: \"3:20\", \"1:02:03\" 형식(초는 두 자리, 영상 길이 이내)을 구간에 연결, 경계에서 {v['comments']['timestamp_tolerance_sec']}초 이내면 가장 가까운 구간.",
        f"말–톤 불일치: 텍스트 감성 {v['mismatch']['text_valence_min']} 이상인데 음성 각성이 {v['mismatch']['audio_arousal_max']} 이하 (음성이 있을 때만).",
    ]


def cx_rules() -> list[str]:
    return [
        "CSAT: 리뷰 별점을 정답 대용으로 13개 특징(평균·최저·마지막 감성, 지수 4개, Peak-End, 불만 수, 이탈·비추천·재주문·추천 의사) 릿지 회귀. 리뷰 이벤트 리뷰는 학습에서 뺀다. 별점 리뷰 20건 이상일 때만 학습.",
        "CSAT 검증: 학습에 쓰지 않은 25%로 MAE를 재고 평균 별점·감성 단일 회귀와 비교. 만족(4점 이상) 기준값은 학습 데이터의 실제 만족 비율에 맞춰 보정.",
        "CES(1 쉬움 ~ 7 어려움) = 1 + 6 × (0.6·좌절/100 + 0.2·반복 문의 + 0.1·대기 + 0.1·보류/이관).",
        "NPS 성향: 추천·재주문 의사 → 추천 고객, 이탈·비추천 의사 → 비추천 고객(둘 다면 비추천), 둘 다 없으면 판단 불가. 판단 가능한 세션만으로 계산.",
        f"채널 비교: 원점수 대신 불만 유형 구성비(채널 안 합계 100%). 조용한 불만 = 다른 채널 비중 10% 이상이면서 리뷰의 2배 이상, 언급 2건 이상. 채널 불만 언급이 {SMALL_CHANNEL_MENTIONS}건 미만이면 표본 적음.",
        "도메인이 다른 데이터(배달·제품)끼리는 비교하지 않는다.",
    ]


# ---------- 실제 데이터 정확도 (experiments/results 의 평가 결과 파일에서) ----------

RESULTS = "experiments/results"
GOLD_MIN_SUPPORT = 4      # 정답 절이 이보다 적은 유형은 표에서 뺀다 (표본이 너무 작다)


def _result(name: str) -> dict | list | None:
    import json

    from voclens.config import REPO_ROOT

    path = REPO_ROOT / RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def accuracy_summary() -> pd.DataFrame:
    """평가별 핵심 수치 한 줄씩. 결과 파일이 없으면 빈 표."""
    rows = []
    old, new = _result("shopping_gold_v0.2.1.json"), _result("shopping_gold_v0.3.0.json")
    if new:
        a = new["any_complaint"]
        rows.append(("불만 여부 (절 단위)", f"네이버 쇼핑 평가 세트 {new['n']}개 절",
                     f"정밀도 {a['precision']:.2f} · 재현율 {a['recall']:.2f} · F1 {a['f1']:.2f}",
                     f"이전 기준 v0.2.1: F1 {old['any_complaint']['f1']:.2f}" if old else ""))
        rows.append(("불만 유형 (절 단위)", f"같은 평가 세트", f"micro F1 {new['micro']['f1']:.2f} · macro F1 {new['macro_f1']:.2f}",
                     f"이전 기준 v0.2.1: micro F1 {old['micro']['f1']:.2f}" if old else ""))
    r_old, r_new = _result("shopping_test_ratings_v0.2.1.json"), _result("shopping_test_ratings_v0.3.0.json")
    if r_new:
        neg = r_new["complaint_rate_by_rating"]
        rows.append(("불만 검출률 (리뷰 단위, 별점 대리 지표)", f"네이버 쇼핑 평가용 {r_new['n']:,}건",
                     f"1점 리뷰 {neg['1']:.0%} · 2점 리뷰 {neg['2']:.0%}에 불만 유형이 붙음",
                     f"이전 기준: 1점 {r_old['complaint_rate_by_rating']['1']:.0%} · 2점 {r_old['complaint_rate_by_rating']['2']:.0%}" if r_old else ""))
        rows.append(("감성 방향 (리뷰 단위)", "같은 데이터", f"고득점(4·5점)·저득점(1·2점) 구분 AUC {r_new['auc_valence_vs_rating']:.2f}", ""))
    kote = _result("kote_test_lexicon.json")
    if kote:
        p = kote["polarity"]
        rows.append(("감정 극성 (문장 단위)", f"KOTE 댓글 {kote['n']:,}문장",
                     f"판단한 문장의 {p['accuracy_when_called']:.0%} 정확 (긍정 {p['accuracy_positive']:.0%}, 부정 {p['accuracy_negative']:.0%})",
                     f"감정 표현이 걸린 문장 {kote['coverage']:.0%}"))
    return pd.DataFrame(rows, columns=["평가", "데이터", "결과", "비교·참고"])


def accuracy_by_type() -> pd.DataFrame:
    """불만 유형별 정밀도·재현율 (평가 세트, 정답 절 GOLD_MIN_SUPPORT개 이상)."""
    new = _result("shopping_gold_v0.3.0.json")
    if not new:
        return pd.DataFrame(columns=["불만 유형", "정답 절", "정밀도", "재현율", "F1", "읽는 법"])
    rows = []
    for label, m in new["per_label"].items():
        if m["support"] < GOLD_MIN_SUPPORT:
            continue
        recall = m["recall"] or 0.0
        note = "잘 잡는 편" if recall >= 0.5 else "절반 넘게 놓침" if recall >= 0.2 else "대부분 놓침 — 결과보다 실제가 많을 수 있음"
        rows.append((label, m["support"], m["precision"], m["recall"], m["f1"], note))
    return pd.DataFrame(rows, columns=["불만 유형", "정답 절", "정밀도", "재현율", "F1", "읽는 법"]).sort_values("정답 절", ascending=False)


def accuracy_notes() -> list[str]:
    return [
        "규칙은 네이버 쇼핑 리뷰의 개발용 80%만 보고 고쳤고, 수치는 규칙을 고칠 때 보지 않은 평가용 20%에서 냈습니다.",
        "평가 세트 정답은 규칙과 예측을 보지 못한 Claude 라벨러 두 개가 붙였습니다(일치도 κ 0.96). 같은 모델이라 사람 둘보다 일치도가 높게 나오며, 사람 검수는 진행 중입니다.",
        "정밀도는 '불만이라고 판단한 것 중 맞은 비율', 재현율은 '실제 불만 중 잡아낸 비율'입니다. 지금 기준선은 정밀도는 높고 재현율은 낮습니다. 결과에 나온 불만은 대체로 믿을 만하지만, 나오지 않은 불만도 있을 수 있습니다.",
        "배달 도메인은 아직 실제 데이터로 정확도를 재지 못했습니다(합성 데이터로만 확인).",
    ]


def domain_reliability(domain: str | None) -> str:
    """직접 분석 결과 옆에 보여 줄 신뢰도 안내 한 줄."""
    new = _result("shopping_gold_v0.3.0.json")
    if domain == "product" and new:
        a = new["any_complaint"]
        return (f"온라인 쇼핑 리뷰 평가 기준: 불만이라고 판단한 문장의 {a['precision']:.0%}가 실제 불만이었고, 실제 불만 문장의 {a['recall']:.0%}를 잡았습니다. "
                "불만 유형 비율은 실제보다 낮게 나올 수 있습니다. 자세한 내용은 '분석 기준 → 정확도' 탭에 있습니다.")
    return "배달 도메인은 실제 데이터로 정확도를 아직 재지 못했습니다. 결과는 사전·규칙 기반 추정치입니다."
