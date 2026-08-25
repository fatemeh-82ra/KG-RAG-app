FROM python:3.12-slim

WORKDIR /app

# System deps (lightweight); pymupdf/chromadb ship manylinux wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT (default 8080 locally)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
