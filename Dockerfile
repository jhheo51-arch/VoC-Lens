# VoC Lens 대시보드 (Hugging Face Spaces Docker, 또는 어떤 컨테이너 호스팅이든)
#   docker build -t voc-lens .
#   docker run -p 7860:7860 voc-lens
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[app]"

COPY configs ./configs
COPY data/samples ./data/samples
COPY app ./app
COPY .streamlit ./.streamlit

# Hugging Face Spaces는 7860 포트를 쓴다. 첫 방문 시 합성 샘플 DB를 자동으로 만든다.
EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]
