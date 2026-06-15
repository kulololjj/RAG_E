# ============================================================================
# 医疗 RAG 系统 - Dockerfile
# ============================================================================
# 构建:  docker build -t medical-rag:latest .
# 运行:  docker run -p 8000:8000 --gpus all medical-rag:latest
# ============================================================================

FROM python:3.11-slim

LABEL maintainer="medical-rag"
LABEL description="医疗 RAG 检索增强生成系统"

# --- 系统依赖 ---
# tesseract-ocr + 中文语言包: OCR 扫描版 PDF
# poppler-utils: pdf2image 需要
# libgl1: OpenCV 间接依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# --- 工作目录 ---
WORKDIR /app

# --- Python 依赖 (使用清华镜像加速) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# --- 应用代码 ---
COPY config.py .
COPY rag_engine.py .
COPY app.py .
COPY api.py .
COPY eval.py .

# --- 目录 ---
RUN mkdir -p /app/bing_pdfs /app/chroma_db /app/my_models

# --- 环境变量 ---
ENV PYTHONUNBUFFERED=1
ENV RAG_LLM_BASE_URL=http://host.docker.internal:11434
ENV RAG_LLM_MODEL=qwen3.5:4b
ENV RAG_RETRIEVAL_K=8
ENV RAG_RERANK_TOP_N=4
ENV RAG_RERANK_ENABLED=false
ENV RAG_PDF_FOLDER=/app/bing_pdfs
ENV RAG_VECTOR_STORE_DIR=/app/chroma_db
ENV RAG_HF_HOME=/app/my_models

# --- 卷 ---
VOLUME ["/app/bing_pdfs", "/app/chroma_db", "/app/my_models"]

# --- 端口 ---
EXPOSE 8000

# --- 启动 ---
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
