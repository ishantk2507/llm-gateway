# syntax=docker/dockerfile:1

# ── Stage 1: dependencies ────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /app

# Deps first — this layer survives source edits.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-install-project --no-dev

# TODO(day-3): pre-bake the embedding model so containers cold-start with
# zero downloads (DESIGN.md §13). Enable once sentence-transformers is in
# pyproject.toml:
#
#   ENV HF_HOME=/opt/hf-cache
#   RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; \
#         S('sentence-transformers/all-MiniLM-L6-v2')"

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:${PATH}"
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
# TODO(day-3): COPY --from=builder /opt/hf-cache /opt/hf-cache   (and set HF_HOME)
# TODO(day-4): COPY data ./data

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY data ./data

EXPOSE 8000
# One worker BY DESIGN — in-memory FAISS index + breaker state (DESIGN.md §13).
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]