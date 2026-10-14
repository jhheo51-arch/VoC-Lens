# 상담 전사본 형식

`voclens call`은 아래 형식의 JSON을 읽는다. 음성 파일만 있으면 `audio.stt`로 전사·화자 분리를 먼저 한다. 이 경로는 `pip install -e .[audio]`와 `HF_TOKEN`이 필요하다.

```json
{
  "call_id": "call_001",
  "meta": {"brand": "배달앱", "domain": "배달", "created_at": "2026-10-02"},
  "audio": "call_001.wav",
  "turns": [
    {"speaker": "A", "start": 0.0, "end": 4.5, "text": "안녕하세요 고객센터 상담원 ○○○입니다"},
    {"speaker": "B", "start": 5.0, "end": 9.0, "text": "주문했는데 콜라가 빠져 있어서요", "stt_conf": 0.92}
  ]
}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `call_id` | 아니오 | 없으면 파일 이름을 쓴다 |
| `meta` | 아니오 | 그대로 DB `sources.meta_json`에 저장된다 |
| `audio` | 아니오 | 16kHz 모노 PCM WAV 권장. JSON 파일 기준 상대 경로 가능. 있으면 음향 특징으로 각성을 추정해 텍스트 감정과 융합한다 |
| `turns[].speaker` | 예 | 화자 분리 라벨 (A, B, SPEAKER_00 등) |
| `turns[].start`, `end` | 예 | 초 단위. `end`는 `start` 이상 |
| `turns[].text` | 예 | 발화 내용. 빈 발화는 버린다 |
| `turns[].role` | 아니오 | `agent` 또는 `customer`. 모든 화자에 있으면 역할 판정을 건너뛴다 |
| `turns[].stt_conf` | 아니오 | STT 신뢰도(0~1). 낮으면 융합에서 텍스트 가중치를 줄인다 |

## 분석 결과

- 통화 단위 (DB `sources.meta_json`)
  - 결과: `resolved`(해결) / `pending`(보류) / `transferred`(이관) / `unknown`
  - 대화 역학: 길이, 고객 발화 점유율, 끼어들기, 최장·총 침묵, 반복 발화 수, 상담원 공감 표현 수
  - 역할 판정 방법과 신뢰도, 음성 사용 여부, SNR
- 지수 6개: 고객 발화만으로 계산. 좌절 지수에는 끼어들기·반복·침묵 가산점을 더한다
- 이벤트 (DB `events`): 고객 Valence·스트레스 급변 구간과 직전 상담원 발화(계기)

## AI Hub 데이터 변환

AI Hub 상담 데이터(98, 100번 등)의 원본 JSON 구조는 데이터를 받은 뒤 확인해서 변환 스크립트(`scripts/convert_aihub_call.py`)를 만든다. 데이터마다 화자 표기와 시간 단위가 달라서 받기 전에 미리 만들지 않는다.
