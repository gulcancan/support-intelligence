# ── Stage 1: Base with system deps ──
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install Python dependencies ──
FROM base AS deps

COPY requirements.txt .
# Install PyTorch CPU-only first (~200MB vs ~2GB for CUDA).
# Using --index-url ensures pip ONLY looks at the CPU wheel index for torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Then install everything else from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Final image ──
FROM deps AS final

# Copy application code
COPY src/ src/
COPY scripts/ scripts/

# Data and models are mounted as volumes, not baked in
# ./data:/app/data (read-only, contains the 300K ticket JSON)
# model_cache:/app/models (persisted across runs)

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
