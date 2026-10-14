# 샘플 데이터

`delivery_reviews_synthetic.csv`는 **실제 리뷰가 아닌 합성 데이터**다.
파이프라인과 리포트를 시연하려고 [scripts/make_synthetic_reviews.py](../../scripts/make_synthetic_reviews.py)로 만들었다.

- 브랜드 이름(치킨브랜드A, 피자브랜드B)과 매장 이름은 가상이다.
- 일부러 넣은 패턴:
  - 치킨브랜드A는 배달 지연·식은 음식 불만이 많다.
  - 피자브랜드B는 누락 불만이 많고, 10월에 누락이 늘어난다.
  - 리뷰 이벤트 리뷰 중에는 불만이 있는데 별점 5점인 경우가 있다.
- 이 데이터로 나온 수치는 모델 성능이 아니다. 성능은 실제 데이터와 라벨로 따로 잰다.

AI Hub 등 외부 데이터는 재배포할 수 없어서 저장소에 넣지 않는다. 신청 방법은 [docs/data_sources.md](../../docs/data_sources.md)에 있다.
