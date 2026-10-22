# 라벨링 기준서 (불만 유형) v0.3

2주차 능동 학습 라벨링에 쓴다. 애매한 사례를 만나면 아래 "판정 사례"에 계속 추가한다.

> **v0.3 (2026-10-22)**: `분류 불가`(이유 없는 부정 평가)를 **불만으로 센다**. v0.2까지는 불만 아님으로 셌다. 사람 검수 50절에서 "다신 안 살게요" 같은 절을 사람은 불만으로, Claude 라벨은 불만 아님으로 봐서 불만 여부가 가장 많이 어긋났기 때문이다([docs/evaluation.md](../docs/evaluation.md) 3절). 유형은 여전히 그 절 안에서만 고르고, 다른 절의 이유를 가져오지 않는다. 새 평가 세트(338절)부터 적용하고, 기존 평가 세트 337절의 정답과 점수는 v0.2 정의 그대로 둔다.

## 작업 단위

- **절 하나**에 라벨을 붙인다. 리뷰 전체가 아니다.
  - 예: "맛은 좋은데 / 배달이 늦어요"는 절 2개로, 각각 라벨을 붙인다.
- 라벨은 여러 개 붙일 수 있다 (다중 라벨).
- 분류 기준: [docs/voc_taxonomy.md](../docs/voc_taxonomy.md)

## 기본 규칙

1. **그 절 안에서 말한 것만 본다.** 앞뒤 절이나 별점으로 추측하지 않는다.
2. **불만의 원인으로 분류한다. 결과로 분류하지 않는다.**
   - "식어서 왔어요"는 온도·식감이다. 배달이 늦어서 식었을 수 있어도, 지연을 직접 말하지 않았으면 배달 지연을 붙이지 않는다.
3. **유형을 붙이지 않는 경우**
   - 긍정 평가는 `칭찬` (불만 아님)
   - 사실 서술, 주문 정보처럼 평가가 없는 절은 `해당 없음` (불만 아님)
   - 부정 평가인데 이 절에 이유가 없거나 어느 분류에도 안 맞으면 `분류 불가`. **불만으로 센다.** 이유가 같은 리뷰의 다른 절에 있어도 이 절에 그 유형을 붙이지 않는다. 그 이유는 그 절에서 따로 라벨된다(같은 불만을 두 번 세지 않기 위해서). 어느 분류에도 안 맞는 이유가 있으면 메모를 남긴다. 분류 기준 개정 후보가 된다.
4. **반어·비꼼**은 실제 의미로 판정한다. "참 빨리도 왔네요 한 시간 만에"는 배달 지연이다.
5. **확신이 없으면** `보류`를 체크하고 넘어간다. 보류 항목은 모아서 한 번에 판정 규칙을 정한다.

## 판정 사례

| 절 | 라벨 | 이유 |
|---|---|---|
| 콜라가 안 왔어요 | 누락·오배송 | 주문 품목 누락 |
| 다른 집 치킨이 온 것 같아요 | 누락·오배송 | 오배송 |
| 소스가 새서 봉투가 엉망이에요 | 포장 상태 | |
| 짜고 식었어요 | 맛, 온도·식감 | 다중 라벨 |
| 요청사항 잘 들어주셨어요 | 칭찬 | 요청사항이라는 단어만으로 불만이 아님 |
| 배달비 포함 3만원이에요 | 해당 없음 | 평가 없음 |
| 가격이 좀 있지만 맛있어요 | 칭찬 | 절 분리가 안 된 경우, 주된 평가로 판정 |
| 사장님 답글이 성의 없어요 | 매장 응대 | |
| 블루를 샀는데 / 블랙이 왔어요 | (앞 절) 해당 없음, (뒤 절) 오배송·누락 | 절 경계에 걸치면 실패를 말한 뒤 절에 붙인다 |
| 한 번 빨았더니 실밥이 풀렸어요 | 품질·마감 | 고장·작동 불량이 아니면 제품 불량 대신 품질·마감 |
| 지난번엔 신선했는데 이번엔 물러요 | 품질 편차, 맛·신선도 | 이전 구매와 비교하면 품질 편차를 함께 붙인다 |
| 가격은 좀 있지만 튼튼해요 | 칭찬 | |
| 2개가 깨져 왔네요 / 다신 주문 하지 않을겁니다 | (앞 절) 포장·파손, (뒤 절) 분류 불가 | 뒤 절은 불만이지만 이유가 앞 절에 있으므로 유형을 가져오지 않는다 |
| 매우 불만족스럽습니다 | 분류 불가 | 불만으로 센다. 이유가 없다 |

## 품질 확인

- 라벨링을 마치고 1주일 뒤, 같은 데이터 100개를 순서를 섞어 다시 라벨링한다.
- 자기 일치도 Cohen's κ를 계산한다. 목표는 0.7 이상이다.
- 일치하지 않은 사례는 위 판정 사례 표에 추가한다.

## 도구 사용법

```bash
# 1. 헷갈리는 절부터 500개 골라 Label Studio 작업 파일 만들기
voclens label prepare data/raw/reviews.csv --n 500 --out labeling/tasks.json
#    임베딩 Zero-shot 유사도 차이로 고르려면 (pip install -e .[nlp])
voclens label prepare data/raw/reviews.csv --n 500 --embed

# 2. Label Studio: labeling/label_studio_config.xml로 프로젝트를 만들고 tasks.json을 가져와 라벨링 → JSON으로 내보내기

# 3. 라벨 가져오기 (기본은 원문 없이 절 ID와 라벨만 저장: AI Hub 데이터 재배포 방지)
voclens label import labeling/export/round1.json --out labeling/export/round1_labels.csv

# 4. 1주일 뒤 같은 절 100개를 순서를 섞어 다시 라벨링하고 자기 일치도 확인
voclens label agreement labeling/export/round1.json labeling/export/round1_recheck.json
```

후보는 규칙 기반 분류가 헷갈리는 순서로 고른다: 여러 유형이 동시에 붙은 절 → 분류 불가 → 주제 표현만으로 붙은 절 → 부정인데 유형이 없는 절 → 중립 → 칭찬 → 확실한 불만. 같은 문장은 한 번만, 한 리뷰에서 2개까지만 고른다.

## 평가 세트 (네이버 쇼핑 리뷰)

위 라벨링은 모델을 **학습**시키기 위한 것이고, 평가 세트는 모델을 **채점**하기 위한 것이다. 섞지 않는다.

- 평가용 20% 리뷰에서 별점마다 50건씩 **무작위**로 뽑은 절 337개다. 헷갈리는 절부터 고르지 않는다.
- 라벨링 화면의 예측 칸은 "(평가용: 숨김)"으로 나온다. 예측을 보고 라벨을 붙이지 않기 위해서다.
- 도메인은 제품이다. 배달 전용 유형(배달 지연, 누락·오배송, 포장 상태 등)이 아니라 제품 유형(배송 지연, 오배송·누락, 포장·파손 등)을 고른다.
- 위의 기본 규칙과 판정 사례를 그대로 따른다. 별점은 보지 않는다(Label Studio 화면에도 없다).
- 불만 여부를 채점할 때 `분류 불가`를 불만으로 셀지는 기준서 버전에 따른다. 기존 337절(v0.2 라벨)은 불만 아님, 새 338절(v0.3 라벨)부터는 불만이다. 유형 지표는 두 버전 모두 `분류 불가`를 유형으로 세지 않는다.
- 유형 정의는 [label_definitions_product.md](label_definitions_product.md)를 쓴다. 분류 규칙의 정규식은 보지 않는다.
- 현재 정답(`labeling/eval/naver_shopping_eval_labels_A.json`, `_B.json`)은 규칙을 보지 못한 Claude 라벨러 두 개가 붙였다. 사람이 일부를 다시 라벨링해 `voclens label agreement`로 일치도를 재는 것이 다음 검증 단계다.

```bash
python scripts/fetch_public_data.py naver_shopping
voclens eval prepare data/external/naver_shopping/naver_shopping.txt        # → data/labeling/eval_tasks.json (원문 포함, 커밋 금지)
# Label Studio에서 labeling/label_studio_config.xml로 프로젝트를 만들고 가져와 라벨링 → data/labeling/eval_export.json으로 내보내기
voclens eval gold data/labeling/eval_export.json --out experiments/results/shopping_gold_v0.3.0.json
```

### 사람 검수 (Claude 라벨 확인)

Label Studio 없이 브라우저에서 라벨링하는 페이지를 쓴다. Claude 라벨은 페이지에 나오지 않는다.

```bash
python scripts/make_labeling_page.py data/labeling/eval_tasks.json --n 50 --out data/labeling/human_check.html
# 브라우저로 열어 라벨링 → "결과 내려받기"로 human_labels.json 저장
voclens label agreement data/labeling/human_labels.json labeling/eval/naver_shopping_eval_labels_A.json
```
