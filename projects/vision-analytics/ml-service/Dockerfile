# ML service, CPU image (default). Small: no PyTorch, only ONNX Runtime. No database (MongoDB is the server's).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY ml-service/requirements.txt .
RUN pip install -r requirements.txt

COPY ml-service/app ./ml-service/app
COPY ml-service/inference ./ml-service/inference
COPY models ./models
COPY training/reports ./training/reports

# non-root user
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser
WORKDIR /app/ml-service
ENV VA_MODELS_DIR=/app/models VA_DATA_DIR=/app/data VA_METRICS_FILE=/app/training/reports/metrics.json
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:8100/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
