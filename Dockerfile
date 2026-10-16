FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/docs ./data/docs

# Pre-build the offline (local embedding) index so the image starts instantly without a key.
# With OPENAI_API_KEY set at runtime, the OpenAI index is built on first start (about 35 chunks,
# a fraction of a cent) and cached in INDEX_DIR.
RUN EMBEDDING_BACKEND=local python scripts/build_index.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
