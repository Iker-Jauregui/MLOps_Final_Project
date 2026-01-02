# Stage 1: Builder (installs dependencies)
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install ONLY production dependencies (excludes dev, train, monitoring)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime (minimal final image)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY api ./api
COPY logic ./logic
COPY templates ./templates
COPY production/model/best_rf.onnx ./production/model/best_rf.onnx
COPY production/model/categorical_metadata.json ./production/model/categorical_metadata.json

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
