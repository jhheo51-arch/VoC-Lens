# 영상 입력 형식

`voclens video`는 **영상 번들 JSON**이나 **YouTube URL**을 받는다.

```bash
voclens video data/samples/videos                 # 번들 폴더
voclens video "https://www.youtube.com/watch?v=..."  # URL (pip install -e .[video])
voclens video "https://youtu.be/..." --stt        # 사람 자막이 없을 때 음성 전사 (pip install -e .[audio])
```

## URL 경로

1. yt-dlp로 메타데이터(제목, 채널, 설명, 태그, 업로드일, 조회수, 좋아요)를 받는다.
2. **사람이 단 한국어 자막**만 받는다. 자동 생성 자막은 정확도가 낮아서 쓰지 않는다.
   - 사람 자막이 없고 `--stt`를 주면 음성을 16kHz WAV로 받아 Whisper로 전사하고 화자를 분리한다. 음성은 임시 폴더와 함께 지운다.
   - 둘 다 안 되면 오류를 낸다. 이때는 번들 JSON을 직접 만든다.
3. `YOUTUBE_API_KEY`가 있으면 YouTube Data API로 댓글을 받는다 (관련도순, 최대 500개). 없으면 댓글 없이 분석한다.

YouTube 이용약관을 지키고 공개 영상만 비상업·연구 목적으로 분석한다. 원본 음성·영상은 보관하지 않는다.

## 번들 JSON

```json
{
  "video_id": "video_001",
  "meta": {
    "title": "무선청소기 한 달 사용 솔직 리뷰",
    "channel": "채널명",
    "description": "...",
    "tags": ["리뷰"],
    "published_at": "2026-09-20",
    "views": 18234,
    "likes": 512,
    "domain": "product",
    "brand": "브랜드명"
  },
  "audio": "선택: video_001.wav",
  "segments": [
    {"start": 0.0, "end": 9.0, "text": "안녕하세요 오늘은 ...", "speaker": "A"}
  ],
  "comments": [
    {"text": "2:00 배터리 부분 공감합니다", "likes": 41, "published_at": "2026-09-21"}
  ]
}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `meta.domain` | 아니오 | `delivery` 또는 `product` (배달·제품도 가능). 다른 도메인 전용 불만 유형을 쓰지 않는다. 예: 제품 영상의 "충전이 오래 걸려"가 배달 지연으로 분류되지 않는다 |
| `meta.views`, `meta.likes` | 아니오 | 있으면 몰입 지수를 시청자 반응(좋아요율, 조회 1천 회당 댓글, 타임스탬프 댓글 비율)으로 계산한다 |
| `segments[].speaker` | 아니오 | 발화 시간이 가장 긴 화자가 크리에이터, 나머지는 게스트 |
| `segments[].text` | 예 | 자막 큐나 전사 문장. 문장으로 다시 나누고 글자 수 비율로 시각을 나눈다 |
| `comments[].text` | 아니오 | "3:20" 같은 시각이 있으면 해당 구간에 연결한다 |
| `audio` | 아니오 | 16kHz 모노 WAV. 있으면 음향 각성을 융합하고 말–톤 불일치를 표시한다 |

## 분석 결과

| 항목 | 저장 위치 | 내용 |
|---|---|---|
| 영상 유형 | `sources.meta_json.video_type` | 리뷰·언박싱 / 브이로그 / 인터뷰 / 상담 재연 / 뉴스 / 기타 |
| 내용 구간 | `units.segment`, `meta_json.segments` | 도입·언박싱·사용·장단점·총평. 구간별 Valence, 주요 속성, 타임스탬프 댓글 수와 감정 |
| 크리에이터 지수 | `indices` | 크리에이터·게스트 문장으로 계산한 6개 지수 |
| 총평 가중 Valence | `meta_json.weighted_valence` | 총평 구간 문장에 가중치 2를 준 평균 |
| 시청자 지수 | `meta_json.audience_indices` | 댓글로 계산한 지수 |
| 크리에이터 vs 시청자 | `meta_json.creator_vs_audience` | 평균 Valence 차이 (양수 = 시청자가 더 긍정적) |
| 감정 급변 | `events` | 크리에이터 Valence 급변 구간 |

구독·좋아요 요청 같은 상투 문구(`configs/video.yaml`의 `boilerplate`)는 분석에서 뺀다.
