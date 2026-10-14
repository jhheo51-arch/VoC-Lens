# 데이터 신청 목록

확인일: 2026-10-14 (AI Hub 사이트에서 데이터셋 번호와 소개 문구를 직접 확인)

AI Hub 데이터는 로그인 후 데이터마다 이용 신청을 해야 한다. 공통 조건은 두 가지다.
- 출처("AI Hub 데이터 사용")를 표시해야 한다.
- 재가공한 데이터를 다시 배포할 수 없다.

## 1. AI Hub — 필수

| 우선 | 용도 | 데이터 | 내용 | 링크 |
|---|---|---|---|---|
| ★ | 상담 음성 + 감정 | 민간분야 고객 상담 데이터 | 교통·금융·보험·생활·의료 분야의 고객 상담 음성 3,300시간. 감정·의도 태그와 요약문이 붙어 있음. **안심존(온라인) 데이터**라 다운로드가 안 되고 AI Hub 원격 환경에서만 분석할 수 있음 | [71616](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71616) |
| ★ | 상담 음성 | 민원(콜센터) 질의-응답 데이터 | 쇼핑(AS·결제·교환·반품·배송·주문) 등 콜센터 상담. 음성 440시간 이상 + 전사 텍스트 | [98](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=98) |
| ★ | 상담 음성 | 상담 음성 | 교육·금융·통신판매 상담센터 녹취 3,000시간 (가상 시나리오 기반) | [100](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=100) |
| ★ | 음성 감정 (SER 학습) | 감정이 태깅된 자유대화 (성인) | 대본 없는 2인 통화 음성. 감정 상태·유형·정도·대상 태그 | [71631](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71631) |
| ★ | 텍스트 감정 | 감성 대화 말뭉치 | 60가지 세부 감정의 대화 말뭉치 27만 문장 + 음성 약 1만 문장 | [86](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86) |
| ★ | 리뷰 ABSA | 속성기반 감정분석 데이터 | 쇼핑몰·SNS 상품 리뷰 약 25만 건, 속성별 감정 태그 | [71603](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71603) |

## 2. AI Hub — 선택 (보완용)

| 용도 | 데이터 | 내용 | 링크 |
|---|---|---|---|
| 상담 음성 (실제 전화 환경) | 복지 분야 콜센터 상담데이터 | 실제 전화망 환경 녹음, 음성 전사 | [470](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=470) |
| 음성 감정 (데이터 추가) | 감정이 태깅된 자유대화 (청소년) | 71631의 청소년판 | [71632](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71632) |
| 상담 텍스트 | 금융분야 고객상담 데이터 | 금융 콜센터 실제 상담 기반 텍스트 | [71926](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71926) |
| 주문 대화 (배달 도메인 어휘) | 소상공인 고객 주문 질의-응답 텍스트 | 소상공인 고객 주문 대화 | [102](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=102) |

## 2-1. 안심존 이용 절차 (71616용)

출처: [AI Hub 안심존 소개](https://www.aihub.or.kr/intrcn/safetyzoneintrcn.do?currMenu=306&topMenu=105) (2026-10-14 확인)

**안심존이란**: 데이터를 밖으로 가져가지 않고 AI Hub의 클라우드 서버 안에서만 분석하는 공간이다.
- 온라인 안심존: 집에서도 보안 네트워크로 접속할 수 있다. 71616이 여기에 해당한다.
- 오프라인 안심존: 지정된 장소에서만 접속할 수 있다.

| 단계 | 내용 | 소요 |
|---|---|---|
| 1 | AI Hub 가입·로그인 → 안심존 포털과 회원 연동(SSO) | |
| 2 | 데이터 페이지의 "이용신청" 버튼 → 안심존 전용 포털로 이동 | |
| 3 | 온라인 신청서 제출. 일반 데이터 필수 서류: **데이터 이용 신청서, 데이터 이용 계획서, 보안서약서** | |
| 4 | NIA가 서류 심의 | 최대 2주 |
| 5 | 승인 후 NIA가 데이터셋·서버를 준비하고 접속 정보 전달 | 이용자가 많으면 대기 |
| 6 | 가상화 서버 접속, 환경 구성 | 데이터 접근 전 최대 5일 |
| 7 | 분석·학습 → 학습한 **모델만** 반출 신청 → NIA 검토 후 전달 | |

**제공 자원 (무료)**
- CPU 서버: 8vCPU, RAM 64GB
- GPU 서버: Tesla V100 32GB
- 저장공간: 작업용 500GB, 모델 반출용 1TB
- 개발 환경: Docker, Jupyter

**프로젝트에 주는 영향**
- 신청부터 데이터 접근까지 **최소 3주 정도** 걸린다 (심의 2주 + 환경 구성 최대 5일 + 대기). 그래서 Day 0에 신청해야 한다.
- 원본 데이터는 반출이 안 되고 **학습한 모델만 반출**할 수 있다. 그래서 71616은 "안심존 안에서 음성 감정·융합 모델을 학습해 모델만 가져오는" 용도로 쓴다. 케이스 스터디의 원문 인용이나 대시보드 데모에는 쓰지 않는다.
- 안심존 GPU(V100)가 Colab 무료 T4보다 좋다. 71616 관련 학습은 안심존에서 한다.
- 안내 페이지에는 "음성데이터 청취 가능"이 오프라인 안심존 항목에만 적혀 있다. **온라인에서 음성을 직접 들을 수 있는지는 확인이 필요하다.** 들을 수 없으면 상담 30건 사람 점수 매기기는 다운로드가 가능한 98번·100번 데이터로 한다.
- 개인 회원이 신청할 수 있는지, 이용 기간과 연장 조건은 안내 페이지에 없다. 신청 전에 AI Hub 문의하기로 확인한다.

## 3. AI Hub 외 데이터

| 용도 | 데이터 | 신청 방법 | 링크 |
|---|---|---|---|
| 음성+텍스트 감정 (융합 학습) | KEMDy20 (ETRI) | 사이트에서 이용 동의서 제출 | [KEMDy20](https://nanum.etri.re.kr/share/kjnoh2/KEMDy20) |
| 음성+텍스트 감정 (추가) | KEMDy19 (ETRI) | 이용 동의서 제출 | [KEMDy19](https://nanum.etri.re.kr/share/kjnoh2/KEMDy19) |
| 온라인 댓글 감정 | KOTE (댓글 5만 개, 43감정 + 감정 없음) | 신청 없이 바로 받을 수 있음 (Hugging Face 표기 라이선스: MIT) | [GitHub](https://github.com/searle-j/KOTE) |
| 쇼핑 리뷰 + 별점 | 네이버 쇼핑 리뷰 공개 데이터 (bab2min/corpus) | 신청 없이 바로 받을 수 있음. 라이선스는 저장소에서 확인 | [GitHub](https://github.com/bab2min/corpus) |
