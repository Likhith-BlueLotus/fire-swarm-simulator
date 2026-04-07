# =============================================================================
# FireSwarm MARL Environment — production Dockerfile
#
# Build context : fire_swarm_simulator/  (the repo root — this file)
# Exposed port  : 7860  (HF Spaces standard)
# Runtime user  : appuser (non-root, required by HF Spaces sandbox)
#
# Base image uses python:3.11-slim-bullseye (explicit Debian codename tag).
# The plain "3.11-slim" tag resolves to a manifest list SHA that is cached
# stale in some build environments; the versioned Debian-codename tag forces
# a fresh layer lookup and avoids the "unexpected status code" registry error.
#
# Build:
#   docker build -t fire-swarm .
#
# Run:
#   docker run -p 7860:7860 \
#     -e API_BASE_URL=https://api.openai.com/v1 \
#     -e MODEL_NAME=gpt-4o-mini               \
#     -e HF_TOKEN=<your-api-key>              \
#     fire-swarm
# =============================================================================

FROM python:3.11-slim-bullseye

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gcc && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

EXPOSE 7860

USER appuser

HEALTHCHECK \
    --interval=15s  \
    --timeout=10s   \
    --start-period=60s \
    --retries=5     \
    CMD python3 /app/healthcheck.py || exit 1

ENV WORKERS=1
ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers ${WORKERS} --log-level info"]
