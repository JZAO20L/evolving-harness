# 构建上下文 = 仓库根（见 compose.yaml 的 context: ../..）
#   docker build -f course/ch20_docker/Dockerfile -t evolving-harness-api:latest .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EH_HOST=0.0.0.0 \
    EH_PORT=8000

WORKDIR /app

# 容器内的 SandboxPool 通过挂载的 docker.sock 调用宿主 daemon，需要 docker CLI。
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "uv>=0.5"

COPY pyproject.toml README.md ./
COPY evolving_harness ./evolving_harness
RUN uv pip install --system ".[dev]"

COPY agents ./agents
COPY course/ch20_docker ./course/ch20_docker

WORKDIR /app/course/ch20_docker
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "backend"]
