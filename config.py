"""
医疗 RAG 系统 - 统一配置文件

使用方法:
    from config import config
    print(config.LLM_BASE_URL)

    # 也可通过环境变量覆盖:
    export RAG_LLM_MODEL="qwen:14b"
"""

import os
from pathlib import Path


class Config:
    """全局配置，支持环境变量覆盖"""

    # ==========================================================================
    # 路径配置
    # ==========================================================================
    BASE_DIR = Path(__file__).parent
    VECTOR_STORE_DIR = str(BASE_DIR / "chroma_db")
    CACHE_INFO_FILE = str(BASE_DIR / "chroma_db" / "cache_info.txt")
    PDF_FOLDER = str(BASE_DIR / "bing_pdfs")
    HF_HOME = str(BASE_DIR / "my_models")

    # ==========================================================================
    # Embedding 模型
    # ==========================================================================
    EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
    EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")  # cpu / cuda

    # ==========================================================================
    # LLM 配置 (Ollama)
    # ==========================================================================
    LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3.5:4b")
    LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", "http://localhost:11434")
    LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))

    # ==========================================================================
    # 文本切分配置
    # ==========================================================================
    CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    CHUNK_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

    # ==========================================================================
    # 检索配置
    # ==========================================================================
    BM25_WEIGHT = float(os.getenv("RAG_BM25_WEIGHT", "0.3"))
    VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "0.7"))
    RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "8"))
    RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "4"))
    RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
    RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-large")
    RERANK_MAX_LENGTH = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))

    # ==========================================================================
    # 对话配置
    # ==========================================================================
    MAX_HISTORY_ROUNDS = int(os.getenv("RAG_MAX_HISTORY_ROUNDS", "4"))
    HISTORY_MAX_ENTRIES = MAX_HISTORY_ROUNDS * 2

    # ==========================================================================
    # OCR 配置 (可选)
    # ==========================================================================
    OCR_ENABLED = os.getenv("RAG_OCR_ENABLED", "false").lower() == "true"

    # ==========================================================================
    # LangSmith 追踪 (可选)
    # ==========================================================================
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "medical-rag")

    @classmethod
    def ensure_dirs(cls):
        """确保必需的目录存在"""
        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(cls.PDF_FOLDER, exist_ok=True)
        os.makedirs(cls.HF_HOME, exist_ok=True)

    @classmethod
    def display(cls):
        """打印当前配置（隐藏敏感信息）"""
        print("=" * 50)
        print("📋 当前配置:")
        print(f"  PDF目录:      {cls.PDF_FOLDER}")
        print(f"  向量库目录:   {cls.VECTOR_STORE_DIR}")
        print(f"  LLM模型:      {cls.LLM_MODEL}")
        print(f"  LLM地址:      {cls.LLM_BASE_URL}")
        print(f"  Embedding:    {cls.EMBEDDING_MODEL}")
        print(f"  分块大小:     {cls.CHUNK_SIZE}")
        print(f"  检索召回数:   {cls.RETRIEVAL_K}")
        print(f"  重排保留数:   {cls.RERANK_TOP_N}")
        print(f"  混合检索:     BM25({cls.BM25_WEIGHT}) + 向量({cls.VECTOR_WEIGHT})")
        print(f"  设备:         {cls.EMBEDDING_DEVICE}")
        print("=" * 50)


# 单例
config = Config()
