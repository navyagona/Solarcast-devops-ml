# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile – SolarCast Region-Based Forecasting API
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="SolarCast Team"
LABEL version="2.0.0"
LABEL description="SolarCast – Region-Based Solar Power Generation Forecasting API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/        ./app/
COPY dashboard/  ./dashboard/
COPY monitoring/ ./monitoring/
COPY model/      ./model/
COPY data/       ./data/
COPY notebooks/  ./notebooks/

RUN addgroup --system solarcast && \
    adduser  --system --ingroup solarcast solarcast && \
    chown -R solarcast:solarcast /app

USER solarcast

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
