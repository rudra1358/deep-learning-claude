# CPU image. The generator (1.5 B LLM) is disabled by default here - on CPU it is too slow for
# interactive use; extractive answers, search, NER, classification and summaries all work.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models-cache DI_ENABLE_GENERATOR=false
WORKDIR /app

COPY backend/requirements.txt .
# CPU-only PyTorch wheel (~200 MB instead of ~2.5 GB for the CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY backend/app ./backend/app
COPY backend/ingest ./backend/ingest
COPY backend/search ./backend/search
COPY backend/nlp ./backend/nlp
COPY models ./models
COPY training/reports ./training/reports

# Download the pretrained models at BUILD time, so containers start fast and work offline.
RUN python -c "from huggingface_hub import snapshot_download as d; \
[d(m, allow_patterns=['*.json','*.safetensors','*.txt','vocab*','merges.txt','*tokenizer*','1_Pooling/*','modules.json']) \
 for m in ['BAAI/bge-small-en-v1.5','cross-encoder/ms-marco-MiniLM-L-6-v2','deepset/roberta-base-squad2']] + [d('sshleifer/distilbart-cnn-12-6', allow_patterns=['*.json','*.txt','pytorch_model.bin'])]"

RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app /models-cache
USER appuser
WORKDIR /app/backend
ENV DI_DATA_DIR=/app/data DI_MODELS_DIR=/app/models DI_METRICS_FILE=/app/training/reports/metrics.json \
    HF_HUB_OFFLINE=1
EXPOSE 8200
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8200"]
