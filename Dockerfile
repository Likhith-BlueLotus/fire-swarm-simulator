# =============================================================================
# FireSwarm MARL Environment — production Dockerfile
#
# Build context : fire_swarm_simulator/  (the repo root — this file)
# Exposed port  : 7860  (HF Spaces standard)
# Runtime user  : appuser (non-root, required by HF Spaces sandbox)
#
# Build:
#   docker build -t fire-swarm .
#
# Run:
#   docker run -p 7860:7860 \
#     -e API_BASE_URL=https://router.huggingface.co/v1 \
#     -e MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct \
#     -e HF_TOKEN=hf_...                                 \
#     fire-swarm
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gcc && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

HEALTHCHECK \
    --interval=30s  \
    --timeout=10s   \
    --start-period=20s \
    --retries=3     \
    CMD python3 /app/healthcheck.py || exit 1

ENV WORKERS=1
ENV PYTHONPATH=/app

CMD uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers ${WORKERS} --log-level info
