# VoiceCX Lens

영상·상담 음성·리뷰에서 고객의 감정을 추정하고, 그 결과를 **우선순위가 매겨진 CX 개선 과제**로 바꾸는 분석 도구입니다.

> 진행 상황: **1주차 — 리뷰 파이프라인 기준선 완료.** 상담 음성·영상 파이프라인은 [구현 계획](docs/implementation_plan.md) 3~5주차에 만듭니다.

## 왜 만들었나

- 기업의 VoC 분석은 대부분 텍스트를 다루지만, 고객의 실제 감정은 목소리 톤과 대화 흐름에 더 잘 드러납니다.
- 감정 점수가 나와도 "그래서 무엇부터 고칠지"로 이어지지 않는 경우가 많습니다.
- 이 프로젝트는 세 채널(영상, 상담 음성, 리뷰)을 **하나의 지수 체계와 불만 분류 기준**으로 분석하고, 결과를 영향도×난이도 매트릭스로 정리합니다.

## 지수 6개

| 지수 | 의미 | 연결되는 CX 지표 |
|---|---|---|
| 스트레스 | 부정 + 높은 각성 (짜증, 분노) | — |
| 행복 | 긍정 (만족, 기쁨) | CSAT |
| 부정 정서 | 부정 + 낮은 각성 (실망, 체념). 의료 진단이 아님 | — |
| 좌절 | 문제를 해결하느라 애쓴 정도 (반복 문의, 대기) | CES |
| 감정 회복 | 부정 감정이 뒤에서 긍정으로 돌아섰는지 | 서비스 회복, FCR |
| 몰입 | 관심과 열의 (구체성, 재주문·추천 의사) | NPS 성향 |

모든 지수는 Valence(긍정–부정) × Arousal(각성) 모델을 바탕으로 합니다. 정의와 산식은 [PRD 7장](docs/PRD.md)과 [`indices/compute.py`](src/voicecx/indices/compute.py)에 있습니다.

## 빠른 시작

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

voicecx review data/samples/delivery_reviews_synthetic.csv
voicecx report --out reports/review_report.md
voicecx query --list
voicecx query complaint_share_by_brand
pytest
```

리뷰 CSV에는 `text` 열이 필요합니다. `rating`, `created_at`, `brand`, `store`, `owner_reply` 열은 있으면 씁니다. 열 이름은 [configs/default.yaml](configs/default.yaml)에서 바꿉니다.

## 결과 예시

합성 데이터 180건으로 만든 [샘플 리포트](docs/sample_report.md)의 일부입니다. 실제 고객 데이터가 아닙니다.

| 순위 | 불만 유형 | 언급 리뷰 | 이탈 의사 (전체 대비) | 추세 | 난이도 | 4분면 |
|---|---|---|---|---|---|---|
| 1 | 누락·오배송 | 13.3% | 33.3% (18.3%) | +33% | 1 | Quick Win |
| 2 | 배달 지연 | 12.8% | 43.5% (18.3%) | -1% | 3 | 핵심 과제 |
| 3 | 온도·식감 | 11.1% | 35.0% (18.3%) | +24% | 2 | 핵심 과제 |

리포트에는 이 밖에 다음 내용이 들어갑니다.
- 브랜드 간 차이 검정 (카이제곱, Mann-Whitney U, 효과 크기)
- 고객 여정 단계별 감정, 속성별 감성
- 별점–감정 불일치 리뷰
- 리뷰 이벤트 편향 점검

## 리뷰 분석 흐름

```
CSV → 정제 (반복 문자, 이모티콘 → 감정 토큰, 개인정보 마스킹, 리뷰 이벤트 표시)
    → 절 분리 ("맛은 좋은데 / 배달이 늦어요")
    → 감정 (7감정 → Valence·Arousal)
    → 불만 유형·속성 (분류 기준 17개 소분류)
    → CX 신호 (재주문·이탈·추천 의사, 반복 문의, 대기, 해결)
    → 고객 여정 단계
    → 지수 6개 + Peak-End + 채널 내 백분위
    → SQLite → SQL 집계 / 개선 과제 리포트
```

지금의 감정·불만 분류는 **사전·규칙 기반 기준선**입니다. 2주차에 능동 학습으로 라벨을 만들고 파인튜닝 모델로 바꿉니다. 그 뒤 기준선과의 성능 비교를 [experiments/](experiments/README.md)에 기록합니다. 분류기 인터페이스는 그대로라서 모델만 바꾸면 됩니다. 임베딩 Zero-shot 분류기와 능동 학습 순위 함수는 이미 [`text/taxonomy.py`](src/voicecx/text/taxonomy.py)에 있습니다.

## 저장소 구조

```
configs/            분류 기준, 감정·CX 신호 사전, 여정 단계, 기본 설정
src/voicecx/
  text/             정제, 절 분리, 감정, 불만 유형, 여정 단계, CX 신호
  indices/          지수 계산
  cx/               통계(축소 추정, 부트스트랩, 검정), 개선 과제 리포트
  store/            SQLite 스키마, SQL 집계 쿼리
  pipelines/        채널별 파이프라인 (현재: 리뷰)
  templates/        리포트 템플릿
labeling/           라벨링 기준서, Label Studio 설정
scripts/            합성 데이터 생성기
docs/               PRD, 분석 설계서, 구현 계획, 데이터 신청 목록, 분류 기준
tests/              단위·통합 테스트
```

## 로드맵

| 주차 | 내용 | 상태 |
|---|---|---|
| 1 | 리뷰 파이프라인 기준선, 분류 기준 v0.1, SQL 집계, 개선 과제 리포트 | 완료 |
| 2 | 능동 학습 라벨링, 감정·ABSA·불만 유형 파인튜닝 (Colab) | 예정 |
| 3 | 상담 음성: STT, 화자 분리·역할 판정, 음성 감정 | 예정 |
| 4 | 대화 흐름 지표, 음성·텍스트 융합, 감정 급변 탐지 | 예정 |
| 5 | 영상: 자막·STT, 구간 분할, 타임스탬프 댓글 연결 | 예정 |
| 6 | CSAT·CES·NPS 추정, 채널 비교, 회복 계기 분석 | 예정 |
| 7 | Streamlit 대시보드, 검증 결과 정리 | 예정 |
| 8 | 케이스 스터디 3건, 배포 | 예정 |

## 데이터와 윤리

- 저장소에는 합성 데이터만 있습니다. AI Hub 등 외부 데이터는 재배포할 수 없어서 넣지 않습니다. 신청 방법은 [데이터 신청 목록](docs/data_sources.md)에 있습니다.
- 배달앱·쇼핑몰은 크롤링하지 않습니다. 리뷰는 정당하게 확보한 CSV나 공개 데이터셋만 씁니다.
- 원문은 개인정보(전화번호, 이메일, 긴 숫자)를 가린 뒤 저장합니다.
- 지수는 추정치입니다. 의료 진단, 채용 평가, 직원 감시, 개인 평가에 쓸 수 없습니다.

## 문서

- [PRD](docs/PRD.md)
- [분석 설계서](docs/analysis_design.md)
- [구현 계획](docs/implementation_plan.md)
- [데이터 신청 목록](docs/data_sources.md)
- [VoC 불만 유형 분류 기준](docs/voc_taxonomy.md)
- [라벨링 기준서](labeling/guidelines.md)
