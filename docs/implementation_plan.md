# 구현 계획: VoiceCX Lens

| 항목 | 내용 |
|---|---|
| 버전 | v0.1 |
| 작성일 | 2026-10-13 |
| 상위 문서 | [PRD.md](PRD.md), [analysis_design.md](analysis_design.md) |
| 기간 | 8주 |
| 환경 | 로컬 Windows 11 (CPU, RAM 16GB) + Google Colab 무료 T4 |

---

## 0. 시작 전 준비 (Day 0, 반나절)

| # | 작업 | 이유 |
|---|---|---|
| 1 | **AI Hub 데이터 신청** — 목록과 링크는 [data_sources.md](data_sources.md) | 승인에 며칠 걸릴 수 있음. 가장 먼저 신청 |
| 2 | KEMDy 데이터 이용 신청 (ETRI) | 동의서 제출이 필요함 |
| 3 | Hugging Face 계정 생성, pyannote 모델 이용 약관 동의, 액세스 토큰 발급 | 화자 분리 모델 다운로드에 필요 |
| 4 | Google Cloud 프로젝트 생성 → YouTube Data API v3 키 발급 | 메타데이터·댓글 수집 |
| 5 | 로컬 설치: Python 3.11, uv, ffmpeg, Git | |
| 6 | GitHub 저장소 생성 (public), `.gitignore`에 `data/`, `.env`, 모델 파일 등록 | AI Hub 데이터는 재배포 불가 |
| 7 | Google Drive에 `voicecx/` 폴더 생성 (Colab 체크포인트·중간 결과 저장) | Colab 세션 끊김 대비 |

API 키와 토큰은 `.env`에만 두고 커밋하지 않는다.

---

## 1. 저장소 구조

```
voicecx-lens/
├── README.md
├── pyproject.toml              # uv로 의존성 관리
├── .env.example                # HF_TOKEN, YOUTUBE_API_KEY 자리만
├── configs/
│   ├── default.yaml            # 경로, 모델 이름, 임계치, 융합 가중치
│   ├── taxonomy.yaml           # VoC 불만 유형 분류 기준 (시드 문장 포함)
│   ├── journey_stages.yaml     # 도메인별 고객 여정 단계
│   ├── lexicons/               # CX 신호 사전 (재주문, 이탈, 반복, 공감, 해결 ...)
│   └── difficulty.yaml         # 불만 유형별 실행 난이도 (개선 과제 리포트용)
├── src/voicecx/
│   ├── schema.py               # 공통 분석 단위 스키마 (pydantic)
│   ├── config.py
│   ├── ingest/                 # 입력 처리
│   │   ├── review_csv.py
│   │   ├── audio_file.py
│   │   └── youtube.py          # 오디오·자막·메타데이터·댓글
│   ├── text/                   # 텍스트 분석 (리뷰·STT 공통)
│   │   ├── preprocess.py       # 정규화, 띄어쓰기, 리뷰 이벤트 표시, 마스킹
│   │   ├── segment.py          # 문장·절 분리
│   │   ├── emotion.py          # 7감정 → V/A
│   │   ├── absa.py
│   │   ├── taxonomy.py         # 불만 유형 (zero-shot / 파인튜닝 모델)
│   │   ├── journey.py
│   │   ├── cx_signals.py
│   │   └── topics.py           # BERTopic 신규 이슈
│   ├── audio/                  # 음성 분석 (상담·영상 공통)
│   │   ├── preprocess.py       # 리샘플, VAD, SNR
│   │   ├── diarize.py          # 화자 분리 + 역할 판정
│   │   ├── stt.py
│   │   ├── acoustic.py         # eGeMAPS + 화자 기준 정규화
│   │   ├── ser.py              # 음성 감정 분류기
│   │   └── dynamics.py         # 대화 역학 지표
│   ├── video/
│   │   ├── classify.py         # 영상 유형
│   │   ├── segment.py          # 내용 구간 분할
│   │   └── comments.py         # 타임스탬프 댓글 연결
│   ├── fusion/
│   │   ├── fuse.py             # 차원별 가중 융합, 신뢰도 기반 조정
│   │   └── calibrate.py        # 온도 스케일링, 백분위 기준 코퍼스
│   ├── indices/
│   │   ├── compute.py          # 6개 지수
│   │   ├── session.py          # Peak-End, 임계치 초과 시간, 종료 값
│   │   └── events.py           # 변화점 탐지
│   ├── cx/
│   │   ├── estimates.py        # CSAT / CES / NPS 추정
│   │   ├── compare.py          # 채널 비교, 조용한 불만
│   │   ├── recovery.py         # 회복 계기 분석
│   │   ├── stats.py            # 베이지안 축소, 부트스트랩, 검정
│   │   └── report.py           # 우선순위 점수, 4분면, 템플릿 리포트
│   ├── store/
│   │   ├── db.py               # SQLite 연결, 마이그레이션
│   │   └── queries/            # 포트폴리오용 SQL 집계 쿼리 (.sql)
│   ├── pipelines/              # 채널별 엔드투엔드 실행
│   │   ├── review.py
│   │   ├── call.py
│   │   └── video.py
│   └── cli.py                  # voicecx review / call / video / report
├── app/
│   └── streamlit_app.py        # 대시보드
├── notebooks/
│   ├── colab/                  # Colab 학습 노트북 (번호순)
│   │   ├── 01_text_emotion_finetune.ipynb
│   │   ├── 02_absa_finetune.ipynb
│   │   ├── 03_taxonomy_finetune.ipynb
│   │   ├── 04_ser_classifier.ipynb
│   │   ├── 05_fusion_stacking.ipynb
│   │   └── 06_batch_stt_demucs.ipynb
│   └── analysis/               # 케이스 스터디 분석
├── labeling/
│   ├── label_studio_config.xml
│   ├── guidelines.md           # 라벨링 기준서
│   └── export/                 # 라벨 결과 (원문 없이 ID + 라벨만 커밋)
├── experiments/                # 실험 결과표, 혼동 행렬, 실패 사례
├── reports/                    # 생성된 리포트 샘플
├── docs/                       # PRD 외 설계 문서, voc_taxonomy.md
├── tests/
│   ├── unit/
│   ├── golden/                 # 작은 고정 입력 → 기대 출력
│   └── fixtures/               # 직접 만든 샘플 리뷰·짧은 녹음
└── data/                       # (gitignore) raw / interim / processed
```

### 설계 결정
- **채널별 파이프라인은 같은 부품을 조립한다.** `text/`는 리뷰, 상담 STT, 영상 자막, 댓글에 모두 쓰인다. 텍스트 모델을 먼저 잘 만들면 나머지 채널이 함께 좋아진다.
- **모델은 교체 가능하게 둔다.** `text/taxonomy.py`는 Zero-shot과 파인튜닝 모델을 같은 인터페이스(`predict(units) -> units`)로 감싼다. 1주차에는 Zero-shot으로 전체 흐름을 먼저 완성하고, 2주차에 파인튜닝 모델로 바꾼다.
- **모든 단계는 `list[Unit]`을 받아 `list[Unit]`을 돌려준다.** 각 단계는 스키마의 필드만 채운다. 단계마다 중간 결과를 Parquet로 캐시해 재실행 시간을 줄인다.
- **학습은 Colab, 추론은 로컬.** 학습한 모델은 Drive → 로컬 `models/`(gitignore)로 내려받는다. 텍스트 모델은 ONNX int8로 변환해서 쓴다.

---

## 2. 데이터 모델

### 2.1 분석 단위 스키마 (`schema.py`)
[analysis_design.md 0.2](analysis_design.md)의 공통 출력 스키마를 pydantic 모델로 구현한다.

### 2.2 DB 테이블 (SQLite)
```sql
sources   (source_id PK, source_type, brand, channel, category, title, url, created_at, meta_json)
sessions  (session_id PK, source_id FK, kind,                       -- 리뷰 1건 / 통화 1건 / 영상 1건
           duration_sec, outcome, peak_end, index_json)
units     (unit_id PK, session_id FK, speaker_role, start_sec, end_sec, text,
           valence, arousal, emotion_json, aspect, aspect_sentiment,
           journey_stage, cx_signals_json, confidence_json)
unit_labels  (unit_id FK, label_type, label, source)                -- source: zero_shot | model | human
indices   (session_id FK, index_name, raw, percentile, mean, peak, above_ratio, end_value)
events    (event_id PK, session_id FK, start_sec, end_sec, index_name, delta, trigger_unit_id)
runs      (run_id PK, pipeline, config_hash, model_versions_json, started_at, status, error)
```
- `unit_labels`에서 사람·모델·Zero-shot 라벨을 구분해 저장한다. 같은 테이블로 학습 데이터 추출과 성능 비교를 함께 한다.
- `runs`에 설정과 모델 버전을 남겨 결과를 재현할 수 있게 한다.
- `store/queries/`에 "브랜드별 불만 유형 비율", "여정 단계 × 지수", "급증 이슈" 같은 SQL을 둔다. 공고의 SQL 역량을 보여주는 증거로 README에 링크한다.

---

## 3. 주차별 계획

각 주차의 **완료 기준**을 만족해야 다음 주로 넘어간다. Colab 표시가 있는 작업은 GPU가 필요하다.

### 1주차: 리뷰 파이프라인 뼈대 + 분류 기준 초안
| # | 작업 | 산출물 |
|---|---|---|
| 1-1 | 저장소 뼈대, `pyproject.toml`, `schema.py`, `config.py`, CLI 골격 | `voicecx --help` 동작 |
| 1-2 | 리뷰 데이터 확보: 네이버 쇼핑 리뷰 공개 데이터 + 배달 리뷰 샘플 CSV (직접 정리) | `data/raw/` |
| 1-3 | `text/preprocess.py`: 반복 문자 정규화, 띄어쓰기, 이모티콘 감정 토큰, 리뷰 이벤트 표시, 개인정보 마스킹 | 단위 테스트 20개 이상 |
| 1-4 | `text/segment.py`: kss 문장 분리 + 대조 표현 절 분리 | 골든 테스트 (리뷰 30개) |
| 1-5 | `text/emotion.py`: 공개 사전학습 감정 모델로 **기준선** 구현 → 7감정·V/A 매핑 | 기준선 성능 기록 |
| 1-6 | `configs/taxonomy.yaml` 초안 (6대분류, 소분류, 시드 문장 각 10~20개) + `docs/voc_taxonomy.md` | 분류 기준 v0.1 |
| 1-7 | `text/taxonomy.py`: 임베딩 유사도 Zero-shot 분류 | 리뷰마다 불만 유형 출력 |
| 1-8 | `text/cx_signals.py`: 사전 기반 재주문·이탈·반복·회복 신호 | 단위 테스트 |
| 1-9 | `pipelines/review.py`: CSV → 지수 → SQLite 저장, 간단한 집계 출력 | **엔드투엔드 1회 성공** |
| 1-10 | `labeling/guidelines.md` 초안 + Label Studio 설치·설정 | 라벨링 준비 완료 |

**완료 기준**: `voicecx review data/raw/sample.csv` 한 줄로 리뷰 1,000건이 5분 안에 DB에 저장되고, 브랜드별 불만 유형 비율이 SQL로 조회된다. (정확도는 아직 기준선 수준이어도 됨)

### 2주차: 텍스트 모델 고도화 (라벨링 + Colab 학습)
| # | 작업 | 산출물 |
|---|---|---|
| 2-1 | Zero-shot 불확실도 기준으로 라벨링 후보 500개 추출 → Label Studio에 불러오기 | 라벨링 대기열 |
| 2-2 | **라벨링 1차** (하루 100~150개 × 4일) | 절 400~500개 라벨 |
| 2-3 | `01_text_emotion_finetune` (Colab): KcELECTRA를 감성 대화 말뭉치·KOTE로 파인튜닝. 문장부호 제거 증강 포함 | 감정 모델 v1 + 기준선 대비 성능표 |
| 2-4 | `02_absa_finetune` (Colab): KLUE-RoBERTa를 속성기반 감정분석 데이터로 학습 | ABSA 모델 v1 |
| 2-5 | `03_taxonomy_finetune` (Colab): 불만 유형 다중 라벨 분류기 | 분류기 v1, Macro F1 |
| 2-6 | ONNX 변환 + int8 양자화, 로컬 추론 속도 측정 | 속도표 |
| 2-7 | `text/journey.py` (같은 방식, 1차는 Zero-shot) | |
| 2-8 | `text/topics.py`: BERTopic으로 "분류 불가" 묶음 분석 → 분류 기준 v0.2 개정 | 개정 이력 기록 |

**완료 기준**: 불만 유형 Macro F1 0.75 이상 (미달이면 오류 분석 후 라벨 보강). 실험 결과가 `experiments/`에 표로 남아 있다.

### 3주차: 음성 파이프라인
| # | 작업 | 산출물 |
|---|---|---|
| 3-1 | `audio/preprocess.py`: 16kHz 모노 변환, 음량 정규화, Silero VAD, SNR 추정 | |
| 3-2 | `audio/stt.py`: faster-whisper (로컬 medium int8), 단어 타임스탬프, 발화별 신뢰도, 도메인 어휘 프롬프트 | 5분 음성 처리 시간 측정 |
| 3-3 | `audio/diarize.py`: pyannote 화자 분리 + 역할 판정 규칙 (인사 스크립트, 정형 표현, 첫 발화자) | 상담 50건 역할 판정 정확도 |
| 3-4 | `audio/acoustic.py`: openSMILE eGeMAPS, 화자별 초반 30초 기준 z-score | |
| 3-5 | `04_ser_classifier` (Colab): emotion2vec+ 임베딩 추출 → AI Hub 감정 음성·KEMDy로 경량 분류기 학습 | SER v1, UA·WA |
| 3-6 | `06_batch_stt_demucs` (Colab): 대량 상담 데이터 STT를 미리 돌려 Drive에 저장 | 검증용 전사본 |

**완료 기준**: 상담 녹음 1건 → 고객/상담원 발화 목록 + 발화별 음향 감정이 나온다. 역할 판정 정확도 95% 이상.

### 4주차: 대화 역학, 융합, 지수
| # | 작업 | 산출물 |
|---|---|---|
| 4-1 | `audio/dynamics.py`: 발화 점유율, 끼어들기, 침묵, 반복 발화(의미 유사도), 공감·해결 표현 | |
| 4-2 | `fusion/fuse.py`: 차원별 가중 융합 + STT 신뢰도·SNR 기반 동적 조정 | |
| 4-3 | `05_fusion_stacking` (Colab): KEMDy로 융합 가중치 학습, 음향 단독·텍스트 단독과 비교 (Ablation) | 융합 효과 표 |
| 4-4 | `fusion/calibrate.py`: 온도 스케일링, 채널별 기준 코퍼스 백분위 | |
| 4-5 | `indices/compute.py`, `session.py`, `events.py`: 6개 지수, Peak-End, 변화점 탐지 | |
| 4-6 | `pipelines/call.py` 완성 + DB 저장 | `voicecx call 파일.wav` |
| 4-7 | **상담 30건 사람 점수 매기기** (좌절·회복, 1~5점) → 지수와 Spearman 상관 | 타당성 검증표 |

**완료 기준**: 상담 1건 → 6개 지수, 이벤트, Peak-End가 DB에 저장된다. 좌절·회복 지수와 사람 점수의 ρ가 0.5 이상 (미달이면 원인 분석을 기록하고 가중치 조정).

### 5주차: 영상 파이프라인
| # | 작업 | 산출물 |
|---|---|---|
| 5-1 | `ingest/youtube.py`: yt-dlp 오디오, 사람이 단 자막, Data API 메타데이터·댓글 | |
| 5-2 | 자막이 있으면 자막 사용, 없으면 STT로 분기. 배경음 분리는 Colab 옵션 | |
| 5-3 | `video/classify.py`: 영상 유형 분류 (메타데이터 + 자막 앞부분) | 30개 영상 정확도 |
| 5-4 | `video/segment.py`: 임베딩 유사도 기반 내용 구간 분할, 구간별 속성·여정 단계 | |
| 5-5 | `video/comments.py`: 댓글을 리뷰 파이프라인으로 분석, 타임스탬프 댓글 타임라인 연결 | |
| 5-6 | 말–톤 불일치 탐지, 총평 구간 가중, 크리에이터 vs 시청자 감정 비교 | |
| 5-7 | `pipelines/video.py` 완성 | `voicecx video URL` |

**완료 기준**: 10분짜리 리뷰 영상 URL 1개 → 구간별 감정, 6개 지수, 댓글 분석이 저장된다 (CPU 기준 20분 이내).

### 6주차: CX 해석 레이어
| # | 작업 | 산출물 |
|---|---|---|
| 6-1 | `cx/stats.py`: 베이지안 축소, 부트스트랩 신뢰구간, 카이제곱·Mann-Whitney + 효과 크기 | 단위 테스트 (알려진 값 대조) |
| 6-2 | `cx/estimates.py`: 별점 회귀로 CSAT 추정 (리뷰 검증셋 MAE), CES·NPS 성향 | 기준선 대비 성능 |
| 6-3 | `cx/compare.py`: 채널 내 표준화, 불만 구성비 비교, 조용한 불만 탐지 | |
| 6-4 | `cx/recovery.py`: 회복 시점 직전 상담원 발화 수집·집계 | 효과적인 응대 표현 Top N |
| 6-5 | `cx/report.py`: 우선순위 점수, 4분면, 근거 원문 선택, Jinja2 템플릿 → Markdown/PDF | 리포트 샘플 |
| 6-6 | 여정 단계 분류기 파인튜닝 (2주차 방식, 라벨 200개 추가) | |
| 6-7 | `store/queries/` SQL 집계 쿼리 10개 정리 | |

**완료 기준**: 리뷰셋 1개로 `voicecx report`를 실행하면 4분면 매트릭스와 근거가 들어간 개선 과제 리포트가 나온다.

### 7주차: 대시보드 + 검증 정리
| # | 작업 | 산출물 |
|---|---|---|
| 7-1 | Streamlit 화면: 홈(전체 요약) / 리뷰 / 상담 / 영상 / 채널 비교 / 리포트 | |
| 7-2 | 타임라인 차트, 이벤트 클릭 시 해당 시각 재생, 여정 히트맵(n<30 흐리게), 신뢰도 표시 | |
| 7-3 | 고지 문구 ("추정치이며 진단·개인 평가 용도 아님") 모든 화면에 고정 | |
| 7-4 | `experiments/` 정리: 모든 모델의 기준선 대비 성능표, 혼동 행렬, 실패 사례 10개 | 검증 리포트 |
| 7-5 | 편향 점검: 성별·연령대별 SER 성능 (데이터 메타정보가 있는 범위에서) | |

**완료 기준**: 처음 보는 사람이 대시보드만 보고 "어떤 이슈가 왜 중요한지" 설명할 수 있다 (지인 1~2명에게 테스트).

### 8주차: 케이스 스터디 + 배포
| # | 작업 | 산출물 |
|---|---|---|
| 8-1 | 케이스 1: 배달 리뷰 브랜드 A vs B | `notebooks/analysis/` + 리포트 |
| 8-2 | 케이스 2: 상담 응대 품질 (좌절·회복, 회복 계기) | 리포트 |
| 8-3 | 케이스 3: 한 브랜드의 채널 통합 분석 (영상 + 댓글 + 리뷰) | 리포트 |
| 8-4 | README: 문제 정의 → 설계 → 결과 → 한계, 공고 매핑표, 아키텍처 그림 | |
| 8-5 | Docker 이미지, Hugging Face Spaces 배포 (데모는 공개 가능한 샘플 데이터만) | 데모 URL |
| 8-6 | 3분 데모 영상 | |

---

## 4. 테스트 전략

| 종류 | 대상 | 방법 |
|---|---|---|
| 단위 테스트 | 전처리, 절 분리, 사전 기반 신호, 통계 함수, 지수 산식, 우선순위 점수 | pytest. 규칙 기반 코드는 테스트를 먼저 쓴다 |
| 골든 테스트 | 채널별 파이프라인 | 직접 만든 작은 입력(리뷰 30개, 30초 녹음 3개)의 출력 스냅샷을 비교. 모델이 바뀌면 차이를 확인하고 갱신 |
| 모델 평가 | 감정, ABSA, 불만 유형, SER, 융합, 역할 판정 | `experiments/`의 평가 스크립트. 고정 시드, 고정 분할 |
| 성능 측정 | 처리 속도 | 리뷰 1,000건, 5분 음성, 10분 영상 기준 시간 기록 |

무거운 모델이 필요한 테스트는 `@pytest.mark.slow`로 분리해, 평소에는 빠른 테스트만 돌린다.

---

## 5. 라벨링 일정

| 시점 | 대상 | 분량 | 예상 시간 |
|---|---|---|---|
| 2주차 | 불만 유형 (절) | 400~500개 | 하루 1~1.5시간 × 4일 |
| 3주차 초 | 불만 유형 자기 일치도 재라벨 | 100개 | 30분 |
| 4주차 | 상담 좌절·회복 점수 | 30건 | 통화당 5~10분, 총 3~5시간 |
| 5주차 | 영상 유형 | 30개 | 1시간 |
| 6주차 | 여정 단계 (절) | 200개 | 2시간 |

라벨링 기준서(`labeling/guidelines.md`)에 애매한 사례와 판정 기준을 계속 추가한다. 이 문서 자체가 분류 기준 수립 역량의 증거가 된다.

---

## 6. 리스크와 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| AI Hub 승인 지연 | 2~4주차 학습 지연 | Day 0에 신청. 기다리는 동안 KOTE, 네이버 리뷰, 공개 영어 SER 데이터로 파이프라인부터 완성 |
| Colab 무료 세션 끊김·GPU 미할당 | 학습 중단 | 에폭마다 Drive에 체크포인트 저장, 이어서 학습. 학습 작업을 1~2시간 단위로 쪼갬 |
| CPU 처리 속도 부족 | 목표 시간 초과 | ONNX int8, Whisper medium, 영상은 자막 우선 사용. 그래도 느리면 데모는 미리 처리한 결과 사용 |
| pyannote Windows 설치 문제 | 화자 분리 지연 | WSL2 또는 Colab에서 화자 분리를 미리 실행하는 방식으로 우회 |
| 음성 감정 성능이 낮음 (한국어 데이터 부족) | 지수 신뢰도 하락 | 융합에서 텍스트 비중을 높이고, 각성 차원에만 음향을 쓰는 방식으로 축소. 한계를 검증 리포트에 공개 |
| 라벨 일관성 부족 | 분류기 성능 정체 | 기준서 보강, κ 측정 후 불일치 사례 재검토 |
| YouTube 자막·댓글 수집 제한 | 영상 분석 일부 불가 | Data API 할당량 안에서 수집, 실패 시 STT만 사용 |
| 일정 지연 | 케이스 스터디 부족 | P1 중 F-15(영상 구간 분할), F-25(신규 이슈), F-45(이슈 알림)를 먼저 미룸. 케이스 1·2를 우선 완성 |

---

## 7. 첫 작업일 체크리스트

다음 작업 세션에서 바로 시작할 순서다.

1. [ ] AI Hub·KEMDy 데이터 신청 (0장 1~2)
2. [ ] HF 토큰, YouTube API 키 발급 → `.env`
3. [ ] 저장소 생성, 1장 구조로 빈 폴더와 `pyproject.toml` 만들기
4. [ ] `schema.py`에 Unit 모델 정의
5. [ ] 샘플 리뷰 CSV 50개 직접 작성 (배달 리뷰 문체, 리뷰 이벤트·대조 표현 포함) → 테스트 픽스처
6. [ ] `text/preprocess.py` 테스트부터 작성 후 구현
