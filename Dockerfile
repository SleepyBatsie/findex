# Stage 1: Builder using official uv image and locked dependencies
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Stage 2: Minimal non-root runtime image
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH="/app/data/index.json" \
    PORT="8000" \
    WORKERS="1"

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
COPY --chown=appuser:appgroup data /app/data

RUN if [ ! -f /app/data/index.json ] && [ -d /app/data ]; then \
      findex index /app/data -o /app/data/index.json --workers 2 --executor serial; \
    fi

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health')"

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
