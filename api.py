"""
医疗 RAG 系统 - FastAPI REST API

启动:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

端点:
    GET  /health              健康检查
    GET  /cache/info           缓存信息
    DELETE /cache              清理缓存
    POST /query                同步问答
    POST /query/stream         流式问答 (SSE)
    POST /documents/upload     上传 PDF (自动增量索引)
    GET  /documents/list       列出已索引的 PDF
"""

import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_engine import (
    add_documents_to_vectorstore,
    clear_cache,
    config,
    detect_pdf_changes,
    init_rag,
    init_rag_streaming,
    load_cached_vectorstore,
    rewrite_query,
    run_incremental_update,
    show_cache_info,
)

# ============================================================================
# 0. 应用生命周期
# ============================================================================
_rag_components = None  # (retriever, streaming_chain)
_qa_chain = None        # fallback RetrievalQA chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 RAG 组件"""
    global _rag_components, _qa_chain
    print("🚀 正在初始化 RAG 系统...")
    config.ensure_dirs()

    try:
        result = init_rag_streaming(use_hybrid=True)
        if result is not None:
            _rag_components = result
            print("✅ 流式 RAG 初始化成功")
        else:
            _qa_chain = init_rag(use_hybrid=True)
            print("⚠️ 流式初始化失败，降级为同步模式")
    except Exception as e:
        print(f"❌ RAG 初始化失败: {e}")

    yield
    print("👋 应用关闭")


app = FastAPI(
    title="医疗 RAG 助手 API",
    description="基于 LangChain + Ollama 的医疗文献检索增强生成系统",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 1. Pydantic 模型
# ============================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    history: List[str] = Field(default_factory=list, description="对话历史")
    use_rewrite: bool = Field(default=True, description="是否启用查询重写")


class SourceDocument(BaseModel):
    index: int
    source: str
    page: int
    preview: str


class QueryResponse(BaseModel):
    question: str
    rewritten_question: Optional[str] = None
    answer: str
    sources: List[SourceDocument] = []
    elapsed_seconds: float


class CacheInfoResponse(BaseModel):
    exists: bool
    is_valid: bool
    path: str
    size_mb: float
    vector_count: Optional[int] = None


class DocumentInfo(BaseModel):
    filename: str
    indexed_at: Optional[str] = None
    chunk_count: Optional[int] = None


class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    chunks_indexed: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    rag_ready: bool
    streaming_available: bool


# ============================================================================
# 2. 依赖注入
# ============================================================================
def get_rag_components():
    """获取 RAG 组件，未初始化时抛错"""
    if _rag_components is None and _qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG 系统尚未初始化完成")
    return _rag_components, _qa_chain


# ============================================================================
# 3. 端点
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    comps, qa = _rag_components, _qa_chain
    return HealthResponse(
        status="healthy" if (comps or qa) else "initializing",
        version="2.0.0",
        model=config.LLM_MODEL,
        rag_ready=(comps is not None or qa is not None),
        streaming_available=(comps is not None),
    )


@app.get("/cache/info", response_model=CacheInfoResponse)
async def get_cache_info():
    """获取缓存信息"""
    import json

    from rag_engine import INDEXED_FILES_PATH, _load_indexed_files

    exists = os.path.exists(config.VECTOR_STORE_DIR)
    is_valid = False
    vector_count = None

    if exists:
        from rag_engine import check_cache_valid

        is_valid = check_cache_valid()
        try:
            embeddings = None  # just to count
            vs, _ = load_cached_vectorstore()
            vector_count = vs._collection.count()
        except Exception:
            pass

    total_size = 0
    if exists:
        for fp in Path(config.VECTOR_STORE_DIR).rglob("*"):
            if fp.is_file():
                total_size += fp.stat().st_size

    return CacheInfoResponse(
        exists=exists,
        is_valid=is_valid,
        path=config.VECTOR_STORE_DIR,
        size_mb=round(total_size / 1024 / 1024, 2),
        vector_count=vector_count,
    )


@app.delete("/cache")
async def delete_cache():
    """清理缓存"""
    try:
        clear_cache()
        return {"success": True, "message": "缓存已清理"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_sync(req: QueryRequest):
    """
    同步问答 - 等待完整回答后一次性返回
    """
    comps, qa = get_rag_components()
    start = datetime.now()

    # 查询重写
    rewritten = req.question
    if req.use_rewrite:
        rewritten = rewrite_query(req.question, req.history)

    # 检索 + 生成
    if comps is not None:
        retriever, chain = comps
        docs = retriever.invoke(rewritten)
        answer = chain.invoke(rewritten)
    elif qa is not None:
        result = qa.invoke({"query": rewritten})
        answer = result["result"]
        docs = result.get("source_documents", [])
    else:
        raise HTTPException(status_code=503, detail="RAG 未就绪")

    # 格式化来源
    sources = []
    for i, doc in enumerate(docs[:10], 1):
        src = doc.metadata.get("source", "未知")
        src = src.replace("\\", "/").split("/")[-1]
        sources.append(
            SourceDocument(
                index=i,
                source=src,
                page=doc.metadata.get("page", 0),
                preview=doc.page_content[:150],
            )
        )

    elapsed = (datetime.now() - start).total_seconds()

    return QueryResponse(
        question=req.question,
        rewritten_question=rewritten if rewritten != req.question else None,
        answer=answer,
        sources=sources,
        elapsed_seconds=round(elapsed, 2),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    流式问答 - Server-Sent Events (SSE) 逐 token 推送

    客户端示例:
        curl -X POST http://localhost:8000/query/stream \
             -H "Content-Type: application/json" \
             -d '{"question": "脂溢性皮炎怎么治疗?"}' \
             --no-buffer
    """
    comps, qa = get_rag_components()

    if comps is None:
        raise HTTPException(
            status_code=400,
            detail="流式输出需要 init_rag_streaming() 成功初始化。请使用 POST /query",
        )

    retriever, chain = comps

    # 查询重写
    rewritten = req.question
    if req.use_rewrite:
        rewritten = rewrite_query(req.question, req.history)

    # 检索文档（先获取来源，在第一个事件中发送）
    docs = retriever.invoke(rewritten)
    sources_data = []
    for i, doc in enumerate(docs[:10], 1):
        src = doc.metadata.get("source", "未知")
        src = src.replace("\\", "/").split("/")[-1]
        sources_data.append(
            {
                "index": i,
                "source": src,
                "page": doc.metadata.get("page", 0),
                "preview": doc.page_content[:150],
            }
        )

    async def event_generator():
        import json

        # 事件1: 元数据（来源文档 + 重写后的查询）
        meta = {
            "type": "meta",
            "rewritten_question": rewritten if rewritten != req.question else None,
            "sources": sources_data,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 事件2+: token 流
        try:
            for chunk in chain.stream(rewritten):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        # 结束事件
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传 PDF 文档并自动增量索引

    curl -X POST http://localhost:8000/documents/upload \
         -F "file=@medical_guide.pdf"
    """
    # 验证文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    comps, qa = get_rag_components()

    # 保存文件
    config.ensure_dirs()
    safe_name = file.filename.replace(" ", "_").replace("/", "_")
    dest_path = os.path.join(config.PDF_FOLDER, safe_name)

    # 避免覆盖：添加时间戳
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(safe_name)
        safe_name = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        dest_path = os.path.join(config.PDF_FOLDER, safe_name)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 增量索引
    chunks_indexed = 0
    try:
        if comps is not None:
            retriever, _ = comps
            # 需要访问 vectorstore 来添加文档
            # retriever 可能是 EnsembleRetriever 或 ContextualCompressionRetriever
            from langchain_community.vectorstores import Chroma

            # 加载 vectorstore 进行增量添加
            vs, _ = load_cached_vectorstore()
            if vs is not None:
                chunks_indexed = add_documents_to_vectorstore([safe_name], vs)
                message = f"上传成功，已索引 {chunks_indexed} 个文本切片"
            else:
                message = "上传成功，但向量数据库加载失败，请重建缓存"
        elif qa is not None:
            message = "上传成功，需重启服务以重建向量库"
        else:
            message = "上传成功，但 RAG 未就绪"
    except Exception as e:
        message = f"上传成功，但增量索引失败: {e}"

    return UploadResponse(
        success=True,
        filename=safe_name,
        message=message,
        chunks_indexed=chunks_indexed,
    )


@app.get("/documents/list", response_model=List[DocumentInfo])
async def list_documents():
    """列出已索引的 PDF 文档"""
    from rag_engine import _load_indexed_files

    indexed = _load_indexed_files()
    result = []

    # 已索引的文件
    for fname, info in indexed.items():
        result.append(
            DocumentInfo(
                filename=fname,
                indexed_at=info.get("indexed_at"),
                chunk_count=info.get("chunk_count"),
            )
        )

    # PDF 目录中的文件（未索引的）
    if os.path.exists(config.PDF_FOLDER):
        indexed_names = set(indexed.keys())
        for f in os.listdir(config.PDF_FOLDER):
            if f.lower().endswith(".pdf") and f not in indexed_names:
                result.append(
                    DocumentInfo(filename=f, indexed_at=None, chunk_count=None)
                )

    return sorted(result, key=lambda x: x.filename)


@app.get("/config")
async def get_config():
    """获取当前配置（不含敏感信息）"""
    return {
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_k": config.RETRIEVAL_K,
        "rerank_top_n": config.RERANK_TOP_N,
        "bm25_weight": config.BM25_WEIGHT,
        "vector_weight": config.VECTOR_WEIGHT,
        "hybrid_search": True,
        "ocr_enabled": config.OCR_ENABLED,
        "pdf_folder": config.PDF_FOLDER,
        "vector_store_dir": config.VECTOR_STORE_DIR,
    }


# ============================================================================
# 4. 启动入口
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
