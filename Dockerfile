FROM python:3.11-slim

# Tesseract for OCR; build tools for native wheels (chromadb deps).
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libtesseract-dev \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV CHROMA_DIR=/app/data/chroma
ENV TESSERACT_CMD=/usr/bin/tesseract

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
