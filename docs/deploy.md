# 대시보드 배포

배포 주소: **https://voc-lens.streamlit.app/** (Streamlit Community Cloud, `main` 브랜치에 push하면 자동으로 다시 배포됨)

대시보드는 첫 방문 시 합성 샘플 데이터(리뷰 180건, 상담 4건, 영상 2건)로 DB를 자동으로 만든다. 그래서 별도 데이터 준비 없이 올릴 수 있다. 외부 데이터(AI Hub 등)는 재배포 금지라 배포본에 넣지 않는다.

## 방법 1. Streamlit Community Cloud (권장, 무료)

1. [share.streamlit.io](https://share.streamlit.io)에 GitHub 계정으로 로그인한다.
2. **Create app** → **Deploy a public app from GitHub**를 고른다.
3. 아래처럼 입력한다.

   | 항목 | 값 |
   |---|---|
   | Repository | `jhheo51-arch/VoC-Lens` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |
   | App URL | 원하는 주소 (예: `voc-lens`) |
   | Advanced settings → Python version | 3.12 |

4. **Deploy**를 누른다. `requirements.txt`의 패키지 설치에 몇 분 걸린다.

저장소가 비공개라면 로그인할 때 Streamlit에 비공개 저장소 접근 권한을 줘야 한다. 포트폴리오로 공개할 거라면 저장소를 공개로 바꾸는 편이 간단하다.

## 방법 2. Hugging Face Spaces (Docker)

1. Hugging Face에서 New Space를 만들고 SDK로 **Docker**를 고른다.
2. 이 저장소 내용을 Space 저장소에 올린다. 루트의 `Dockerfile`이 7860 포트로 대시보드를 띄운다.
3. Space 저장소의 `README.md` 맨 위에 다음 설정을 넣는다.

   ```yaml
   ---
   title: VoC Lens
   sdk: docker
   app_port: 7860
   ---
   ```

## 관리 모드

공개 배포에서는 사이드바의 DB 경로 입력과 "합성 샘플로 다시 분석" 버튼을 숨긴다. 방문자가 서버의 임의 경로에 파일을 만들거나 다른 방문자가 보는 데이터를 지울 수 없게 하기 위해서다. 로컬에서 이 기능이 필요하면 환경 변수를 켜고 실행한다.

```bash
VOCLENS_DASHBOARD_ADMIN=1 streamlit run app/streamlit_app.py
```

## 로컬에서 배포본 확인

```bash
docker build -t voc-lens .
docker run -p 7860:7860 voc-lens
```

브라우저에서 `http://localhost:7860`을 연다.
