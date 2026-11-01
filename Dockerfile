FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY Frontend/frontend/package.json Frontend/frontend/package-lock.json ./
RUN npm ci
COPY Frontend/frontend/ ./
ARG VITE_API_URL=/api/
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

COPY Backend/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY Backend/main.py ./main.py
COPY Backend/Agents/ ./Agents/
COPY Backend/Common/ ./Common/
COPY Backend/Controllers/ ./Controllers/
COPY Backend/Routes/ ./Routes/
COPY Backend/Retreival/HybridSearch.py Backend/Retreival/KeywordSearch.py Backend/Retreival/SemanticSearch.py ./Retreival/
COPY --from=frontend-build /frontend/dist/ ./static/

RUN mkdir -p /app/Collections /home/app/.cache \
    && chown -R app:app /app /home/app

USER app

RUN python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-multilingual-cased')" \
    && python -c "import chromadb; chromadb.utils.embedding_functions.DefaultEmbeddingFunction()(['model warmup'])"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
