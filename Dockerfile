# FastAPI service image (Cloud Run).
#
# Build:  docker build -t sec-rag-api .
# Secrets are NEVER baked in — they are passed as Cloud Run env vars at deploy
# time (OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL, SEC_RAG_API_KEY).
# Cloud Run injects $PORT; uvicorn binds to it.

FROM python:3.11-slim

# Faster, quieter, no .pyc clutter.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps from the committed lockfile first (layer caches unless it changes).
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# App source + the config the service reads at startup. README is copied because
# pyproject declares it as the package readme (hatchling reads it at build time).
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

EXPOSE 8080
# $PORT is set by Cloud Run (defaults to 8080); shell form expands it.
CMD exec uvicorn sec_rag.api.app:app --host 0.0.0.0 --port ${PORT:-8080}
