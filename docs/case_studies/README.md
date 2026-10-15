# 케이스 스터디

VoC Lens로 세 가지 CX 질문에 답한 사례입니다.

> **모든 사례는 합성 데이터로 만든 시연용입니다.** 브랜드·매장·채널 이름은 가상이고, 데이터에는 일부러 넣은 패턴이 있습니다([data/samples](../../data/samples/README.md)). 목적은 "도구가 이런 질문에 이런 근거로 답할 수 있다"를 보여주는 것입니다. 실제 고객 데이터의 결론이나 모델 성능이 아닙니다. AI Hub 데이터 승인 후 같은 형식으로 실제 데이터 사례를 추가합니다.

| # | 질문 | 채널 | 핵심 방법 |
|---|---|---|---|
| [1](01_delivery_brand_comparison.md) | 두 배달 브랜드는 무엇을 먼저 고쳐야 하나? | 리뷰 180건 | 불만 유형 분류, 우선순위 점수, 브랜드 간 검정, 리뷰 이벤트 편향 |
| [2](02_call_quality_recovery.md) | 어떤 상담이 고객을 지치게 하고, 무엇이 감정을 회복시키나? | 상담 4건 | 대화 역학, 좌절·감정 회복 지수, Peak-End, 회복 계기 |
| [3](03_cross_channel_chicken_a.md) | 리뷰만 보면 놓치는 것이 있나? | 리뷰 90건 + 영상 1건 + 댓글 7개 | 채널별 불만 구성비, 영상 구간 × 타임스탬프 댓글 |

## 재현

```bash
pip install -e ".[app]"
python -c "from voclens.config import load_settings; from voclens.dashboard.data import build_sample_db; build_sample_db('data/voclens.db', load_settings())"
voclens report --brand 피자브랜드B
voclens query call_overview
voclens cx --brand 치킨브랜드A
streamlit run app/streamlit_app.py
```

수치는 2026-10-15 기준 코드(분류 기준 v0.2.1, 사전 기반 기준선 모델)로 만든 값입니다. 모델을 바꾸면 수치도 달라집니다.
