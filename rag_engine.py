"""
医疗 RAG 引擎 - 核心模块
整合了 PDF 处理、向量数据库、混合检索、重排序、查询重写等全部功能。

使用方式:
    from rag_engine import init_rag, rewrite_query, show_cache_info, clear_cache

    qa_chain = init_rag(use_hybrid=True)
    result = qa_chain.invoke("脂溢性皮炎怎么治疗?")
"""

import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- 环境变量必须在导入 LangChain 之前设置 ---
from config import config

os.environ["HF_HOME"] = config.HF_HOME
os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # 强制离线，不联网
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


# ============================================================================
# 0. LLM 工厂 — 根据 config 创建对应 provider 的 LLM 实例
# ============================================================================
def _create_llm(temperature: Optional[float] = None) -> object:
    """
    创建 LLM 实例，provider 由环境变量 RAG_LLM_PROVIDER 决定。

    Provider → 对应类:
        ollama    → ChatOllama      (本地，默认)
        openai    → ChatOpenAI      (需 RAG_LLM_API_KEY)
        anthropic → ChatAnthropic   (需 RAG_LLM_API_KEY)
        deepseek  → ChatOpenAI      (兼容 OpenAI 协议，base_url 指向 DeepSeek)
    """
    provider = config.LLM_PROVIDER.lower()
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    if provider == "ollama":
        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=temp,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY or os.getenv("OPENAI_API_KEY", ""),
            temperature=temp,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY or os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=temp,
        )
    elif provider == "deepseek":
        ds_url = config.LLM_BASE_URL
        if provider != config.LLM_PROVIDER or "localhost" in ds_url:
            ds_url = "https://api.deepseek.com/v1"
        return ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=ds_url,
            temperature=temp,
        )
    else:
        logger.error(f"未知 LLM provider: {provider}，回退到 Ollama")
        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=temp,
        )


# --- 日志配置 ---
logger = logging.getLogger("medical_rag")
logger.setLevel(logging.DEBUG)

# 控制台输出
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_ch)

# 文件输出
_fh = logging.FileHandler("rag.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(_fh)

# --- 全局变量 ---
cached_documents: Optional[List[Document]] = None  # 缓存原始文档供 BM25 使用


# ============================================================================
# 1. OCR 文档加载器
# ============================================================================
class OCRDocumentLoader:
    """
    OCR 文档加载器 - 自动检测并处理扫描版 PDF

    工作流程:
        1. 先用 PyPDFLoader 尝试直接提取文字
        2. 如果文字过少 (< 50 字符/第一页)，自动触发 OCR
        3. OCR 使用 pytesseract + pdf2image (免费本地方案)
        4. 如果 OCR 也不可用，降级返回 PyPDFLoader 结果
    """

    @staticmethod
    def _ocr_single_page(image, page_num: int) -> str:
        """对单页图像执行 OCR"""
        try:
            import pytesseract

            # 中英文混合识别
            text = pytesseract.image_to_string(
                image, lang="chi_sim+eng", config="--psm 6"
            )
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR 第{page_num}页失败: {e}")
            return ""

    @staticmethod
    def _load_with_ocr(pdf_path: str) -> List[Document]:
        """使用 OCR 完整处理 PDF"""
        from langchain_core.documents import Document as LCDocument

        logger.info(f"🔍 OCR 处理: {os.path.basename(pdf_path)}")

        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=300)
            logger.info(f"  转换 {len(images)} 页为图像")

            documents = []
            for i, image in enumerate(images, 1):
                text = OCRDocumentLoader._ocr_single_page(image, i)
                if text:
                    documents.append(
                        LCDocument(
                            page_content=text,
                            metadata={
                                "source": pdf_path,
                                "page": i,
                                "ocr": True,
                            },
                        )
                    )

            # 如果 OCR 结果也很少，降低 DPI 重试
            total_chars = sum(len(d.page_content) for d in documents)
            if total_chars < 100 and len(images) > 0:
                logger.info("  文字较少，尝试低 DPI ...")
                images_low = convert_from_path(pdf_path, dpi=150)
                documents = []
                for i, image in enumerate(images_low, 1):
                    text = OCRDocumentLoader._ocr_single_page(image, i)
                    if text:
                        documents.append(
                            LCDocument(
                                page_content=text,
                                metadata={
                                    "source": pdf_path,
                                    "page": i,
                                    "ocr": True,
                                },
                            )
                        )

            logger.info(
                f"  OCR 完成: {len(documents)} 页, "
                f"共 {sum(len(d.page_content) for d in documents)} 字符"
            )
            return documents

        except ImportError as e:
            logger.error(
                f"OCR 依赖缺失: {e}。安装: pip install pytesseract pdf2image; "
                f"apt install tesseract-ocr tesseract-ocr-chi-sim poppler-utils"
            )
            return []
        except Exception as e:
            logger.error(f"OCR 处理失败: {e}")
            return []

    @staticmethod
    def load_pdf_with_ocr(pdf_path: str, use_ocr: bool = False) -> List[Document]:
        """
        加载 PDF 文档，自动检测是否需要 OCR

        Args:
            pdf_path: PDF 文件路径
            use_ocr: 强制使用 OCR（跳过自动检测）

        Returns:
            List[Document]: 文档对象列表
        """
        # 如果强制 OCR，直接走 OCR 流程
        if use_ocr:
            ocr_docs = OCRDocumentLoader._load_with_ocr(pdf_path)
            if ocr_docs:
                return ocr_docs
            # OCR 失败则降级
            logger.warning("OCR 失败，降级为普通加载")
            try:
                return PyPDFLoader(pdf_path).load()
            except Exception:
                return []

        # 自动检测模式：先尝试普通加载
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                logger.info(
                    f"{os.path.basename(pdf_path)} 无文字，触发 OCR"
                )
                return OCRDocumentLoader._load_with_ocr(pdf_path)

            # 检测第一页的文字量
            first_page_text = documents[0].page_content.strip() if documents else ""
            if len(first_page_text) < 50:
                logger.info(
                    f"{os.path.basename(pdf_path)} 文字较少"
                    f"({len(first_page_text)} 字符/第一页)，触发 OCR"
                )
                ocr_result = OCRDocumentLoader._load_with_ocr(pdf_path)
                if ocr_result:
                    return ocr_result

            return documents

        except Exception as e:
            logger.warning(f"普通加载失败: {e}，尝试 OCR ...")
            ocr_result = OCRDocumentLoader._load_with_ocr(pdf_path)
            if ocr_result:
                return ocr_result
            # 最终降级
            logger.warning("所有加载方式均失败，返回空列表")
            return []


# ============================================================================
# 2. 缓存管理
# ============================================================================
def get_pdf_hash() -> Optional[str]:
    """计算 PDF 文件夹的哈希值，用于检测文件变化"""
    if not os.path.exists(config.PDF_FOLDER):
        return None

    pdf_files = [
        f for f in os.listdir(config.PDF_FOLDER) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        return None

    hasher = hashlib.md5()
    for pdf_file in sorted(pdf_files):
        file_path = os.path.join(config.PDF_FOLDER, pdf_file)
        if os.path.exists(file_path):
            file_stat = os.stat(file_path)
            hasher.update(
                f"{pdf_file}:{file_stat.st_size}:{file_stat.st_mtime}".encode()
            )

    return hasher.hexdigest()


def check_cache_valid() -> bool:
    """检查缓存是否有效"""
    if not os.path.exists(config.VECTOR_STORE_DIR):
        return False
    if not os.path.exists(os.path.join(config.VECTOR_STORE_DIR, "index")):
        return False
    if not os.path.exists(config.CACHE_INFO_FILE):
        return False

    with open(config.CACHE_INFO_FILE, "r") as f:
        saved_hash = f.read().strip()

    current_hash = get_pdf_hash()
    if current_hash is None:
        return False

    return saved_hash == current_hash


def save_cache_info():
    """保存缓存信息"""
    current_hash = get_pdf_hash()
    if current_hash:
        with open(config.CACHE_INFO_FILE, "w") as f:
            f.write(current_hash)


def show_cache_info():
    """显示缓存信息"""
    print("\n" + "=" * 50)
    print("📊 缓存信息:")
    if os.path.exists(config.VECTOR_STORE_DIR):
        total_size = 0
        for file_path in Path(config.VECTOR_STORE_DIR).rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        print(f"  - 向量数据库路径: {config.VECTOR_STORE_DIR}")
        print(f"  - 数据库大小: {total_size / 1024 / 1024:.2f} MB")
        if check_cache_valid():
            print("  - 缓存状态: ✅ 有效")
        else:
            print("  - 缓存状态: ⚠️ 无效或不存在")
    else:
        print("  - 缓存状态: ❌ 未找到缓存")
    print("=" * 50)


def clear_cache():
    """清理缓存"""
    confirm = input("\n⚠️ 确定要清理缓存吗？这将需要重新构建向量数据库 (y/n): ")
    if confirm.lower() == "y":
        if os.path.exists(config.VECTOR_STORE_DIR):
            shutil.rmtree(config.VECTOR_STORE_DIR)
            logger.info("缓存已清理")
            print("✅ 缓存已清理")
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)


INDEXED_FILES_PATH = os.path.join(config.VECTOR_STORE_DIR, "indexed_files.json")


def _get_file_hash(filepath: str) -> str:
    """计算单个文件的 MD5 哈希"""
    hasher = hashlib.md5()
    stat = os.stat(filepath)
    hasher.update(f"{stat.st_size}:{stat.st_mtime}".encode())
    return hasher.hexdigest()


def _load_indexed_files() -> Dict[str, dict]:
    """加载已索引文件记录"""
    if not os.path.exists(INDEXED_FILES_PATH):
        return {}
    try:
        with open(INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_indexed_files(records: Dict[str, dict]):
    """保存已索引文件记录"""
    with open(INDEXED_FILES_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def detect_pdf_changes() -> Tuple[List[str], List[str], List[str]]:
    """
    检测 PDF 文件变化

    Returns:
        Tuple: (new_files, modified_files, deleted_files)
    """
    if not os.path.exists(config.PDF_FOLDER):
        return [], [], []

    current_pdfs = {
        f for f in os.listdir(config.PDF_FOLDER) if f.lower().endswith(".pdf")
    }
    indexed = _load_indexed_files()

    new_files = []
    modified_files = []
    deleted_files = []

    for pdf_file in current_pdfs:
        if pdf_file not in indexed:
            new_files.append(pdf_file)
        else:
            current_hash = _get_file_hash(os.path.join(config.PDF_FOLDER, pdf_file))
            if current_hash != indexed[pdf_file].get("hash"):
                modified_files.append(pdf_file)

    for indexed_file in indexed:
        if indexed_file not in current_pdfs:
            deleted_files.append(indexed_file)

    return new_files, modified_files, deleted_files


def add_documents_to_vectorstore(
    pdf_files: List[str],
    vectorstore: Chroma,
) -> int:
    """
    增量添加新的 PDF 文档到已有向量数据库

    Args:
        pdf_files: 要添加的 PDF 文件名列表
        vectorstore: 已有的向量数据库

    Returns:
        int: 新增的文本切片数量
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )

    all_texts = []
    indexed = _load_indexed_files()

    for pdf_file in pdf_files:
        full_path = os.path.join(config.PDF_FOLDER, pdf_file)
        logger.info(f"增量索引: {pdf_file}")

        try:
            loader = PyPDFLoader(full_path)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)

            doc_texts = [t.page_content for t in texts]
            doc_metadatas = [t.metadata for t in texts]

            # 添加到向量数据库
            vectorstore.add_texts(
                texts=doc_texts,
                metadatas=doc_metadatas,
            )

            # 更新全局文档缓存
            if cached_documents is not None:
                cached_documents.extend(documents)

            # 记录索引状态
            indexed[pdf_file] = {
                "hash": _get_file_hash(full_path),
                "chunk_count": len(texts),
                "indexed_at": datetime.now().isoformat(),
            }

            all_texts.extend(texts)
            logger.info(f"  ✅ 添加 {len(texts)} 个切片")

        except Exception as e:
            logger.error(f"  ❌ 增量索引失败 {pdf_file}: {e}")

    # 持久化
    vectorstore.persist()
    _save_indexed_files(indexed)

    return len(all_texts)


def remove_documents_from_vectorstore(
    pdf_files: List[str],
    vectorstore: Chroma,
):
    """
    从向量数据库中删除指定 PDF 的文档

    Args:
        pdf_files: 要删除的 PDF 文件名列表
        vectorstore: 向量数据库
    """
    indexed = _load_indexed_files()

    for pdf_file in pdf_files:
        logger.info(f"移除索引: {pdf_file}")
        try:
            # ChromaDB 按 metadata 过滤删除
            vectorstore._collection.delete(
                where={"source": os.path.join(config.PDF_FOLDER, pdf_file)}
            )
            indexed.pop(pdf_file, None)
            logger.info(f"  ✅ 已移除 {pdf_file}")
        except Exception as e:
            logger.warning(f"  ⚠️ 移除 {pdf_file} 失败: {e}")

    vectorstore.persist()
    _save_indexed_files(indexed)


def run_incremental_update(vectorstore: Chroma) -> Dict:
    """
    运行增量更新：检测变化并更新向量数据库

    Returns:
        Dict: 更新摘要
    """
    new_files, modified_files, deleted_files = detect_pdf_changes()

    if not new_files and not modified_files and not deleted_files:
        logger.info("未检测到 PDF 变化，无需更新")
        return {"new": 0, "modified": 0, "deleted": 0, "chunks_added": 0}

    summary = {
        "new": len(new_files),
        "modified": len(modified_files),
        "deleted": len(deleted_files),
        "chunks_added": 0,
    }

    # 删除已移除的 PDF
    if deleted_files:
        remove_documents_from_vectorstore(deleted_files, vectorstore)

    # 增量添加新文件和修改过的文件
    changed_files = new_files + modified_files
    if changed_files:
        # 对于修改过的文件，先删旧数据再添加
        if modified_files:
            remove_documents_from_vectorstore(modified_files, vectorstore)
        chunks = add_documents_to_vectorstore(changed_files, vectorstore)
        summary["chunks_added"] = chunks

    return summary


# ============================================================================
# 3. 查询重写
# ============================================================================
def rewrite_query(user_query: str, history: List[str], llm: object = None) -> str:
    """
    增强版查询重写 - 处理多轮对话中的指代消解

    Args:
        user_query: 用户当前提问
        history: 对话历史列表
        llm: 可选的 LLM 实例（不传则用全局配置创建）

    Returns:
        str: 重写后的完整问题
    """
    if not history:
        return user_query

    recent = history[-config.HISTORY_MAX_ENTRIES :] if len(history) > config.HISTORY_MAX_ENTRIES else history
    history_str = "\n".join(recent)

    rewrite_prompt = f"""你是一个医疗问答系统的查询重写助手。

任务：根据对话历史，将用户的最新提问改写成一个**独立、完整、包含所有必要上下文**的问题。

重写规则：
1. 把"它"、"这个"、"那个"、"这种"等代词替换成具体医学术语
2. 把"上面提到的"、"之前的"、"前面说的"等指代补充完整
3. 如果是连续追问，把前文的诊断/药物名称融入问题
4. 保持医学术语的准确性（如"脂溢性皮炎"不要简化为"皮炎"）
5. 如果问题已经很完整，直接原样输出

对话历史：
{history_str}

最新提问：{user_query}

请直接输出重写后的完整问题（不要加任何解释，不要加引号）："""

    try:
        temp_llm = llm if llm is not None else _create_llm(temperature=0.1)
        response = temp_llm.invoke(rewrite_prompt)
        rewritten = response.content.strip().strip("\"'")

        if rewritten != user_query:
            logger.info(f"查询重写: '{user_query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"查询重写失败: {e}，使用原始查询")
        return user_query


# ============================================================================
# 4. PDF 处理与向量数据库构建
# ============================================================================
def _create_embeddings() -> HuggingFaceEmbeddings:
    """创建 Embedding 模型实例"""
    model_kwargs = {
        "device": config.EMBEDDING_DEVICE,
        "local_files_only": True,
    }
    # 找到本地 snapshot 路径，直接用绝对路径加载，绕开 HF 缓存
    hub_dir = os.path.join(config.HF_HOME, "hub")
    model_dir_name = f"models--{config.EMBEDDING_MODEL.replace('/', '--')}"
    snapshots_dir = os.path.join(hub_dir, model_dir_name, "snapshots")
    if os.path.isdir(snapshots_dir):
        subdirs = os.listdir(snapshots_dir)
        if subdirs:
            local_model_path = os.path.join(snapshots_dir, subdirs[0])
            logger.info(f"从本地路径加载 embedding: {local_model_path}")
            return HuggingFaceEmbeddings(
                model_name=local_model_path,
                cache_folder=os.path.join(config.HF_HOME, "hub"),
                model_kwargs=model_kwargs,
            )
    # 回退
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=os.path.join(config.HF_HOME, "hub"),
        model_kwargs=model_kwargs,
    )


def process_pdfs_and_build_vectorstore(
    use_ocr: bool = False,
) -> Tuple[Optional[Chroma], Optional[HuggingFaceEmbeddings], Optional[List[Document]]]:
    """
    处理 PDF 并构建向量数据库

    Args:
        use_ocr: 是否使用 OCR 处理扫描版 PDF

    Returns:
        Tuple: (vectorstore, embeddings, all_documents)
    """
    global cached_documents

    logger.info("正在扫描 PDF 文件...")
    if not os.path.exists(config.PDF_FOLDER):
        logger.error(f"PDF 目录不存在: {config.PDF_FOLDER}")
        return None, None, None

    pdf_files = [
        f for f in os.listdir(config.PDF_FOLDER) if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        logger.warning("没找到 PDF 文件！")
        return None, None, None

    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )

    all_documents: List[Document] = []
    all_texts: List[Document] = []

    logger.info(f"准备处理 {len(pdf_files)} 个文件...")
    process_start = datetime.now()

    for i, pdf_file in enumerate(pdf_files, 1):
        full_path = os.path.join(config.PDF_FOLDER, pdf_file)
        logger.info(f"[{i}/{len(pdf_files)}] 正在处理：{pdf_file}")

        try:
            documents = OCRDocumentLoader.load_pdf_with_ocr(full_path, use_ocr=use_ocr)
            if not documents:
                logger.warning(f"无法读取 {pdf_file}，跳过")
                continue

            all_documents.extend(documents)
            texts = text_splitter.split_documents(documents)
            all_texts.extend(texts)
            logger.debug(f"  提取 {len(texts)} 个文本切片")

        except Exception as e:
            logger.error(f"处理失败 {pdf_file}: {e}")

    process_time = datetime.now() - process_start
    logger.info(
        f"PDF 处理完成！耗时 {process_time.total_seconds():.2f} 秒，"
        f"共 {len(all_documents)} 个文档，{len(all_texts)} 个文本切片"
    )

    if not all_texts:
        logger.warning("文本切片数量为 0！请检查 PDF 文件是否包含可复制的文字。")
        return None, None, None

    # 调试：打印第一个切片
    logger.debug(f"第一个切片预览: {all_texts[0].page_content[:100]}...")

    # 准备向量化数据
    doc_texts = [t.page_content for t in all_texts]
    doc_metadatas = [t.metadata for t in all_texts]

    # 构建向量数据库
    logger.info("正在构建向量知识库...")
    embed_start = datetime.now()

    embeddings = _create_embeddings()
    vectorstore = Chroma.from_texts(
        texts=doc_texts,
        embedding=embeddings,
        metadatas=doc_metadatas,
        persist_directory=config.VECTOR_STORE_DIR,
    )
    vectorstore.persist()

    embed_time = datetime.now() - embed_start
    logger.info(f"向量数据库构建完成！耗时 {embed_time.total_seconds():.2f} 秒")

    save_cache_info()
    cached_documents = all_documents

    return vectorstore, embeddings, all_documents


# ============================================================================
# 5. 从缓存加载
# ============================================================================
def load_cached_vectorstore() -> Tuple[Optional[Chroma], Optional[HuggingFaceEmbeddings]]:
    """从缓存加载向量数据库"""
    logger.info("从缓存加载向量数据库...")

    embeddings = _create_embeddings()
    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_DIR,
        embedding_function=embeddings,
    )

    try:
        count = vectorstore._collection.count()
        logger.info(f"成功加载向量数据库，包含 {count} 个向量")
    except Exception:
        logger.info("成功加载向量数据库")

    return vectorstore, embeddings


# ============================================================================
# 6. 混合检索器 + 重排序
# ============================================================================
def create_hybrid_retriever_with_rerank(
    vectorstore: Chroma,
    documents: List[Document],
) -> ContextualCompressionRetriever:
    """
    创建混合检索器 (BM25 + 向量) + 重排序

    工作流程:
        1. BM25 关键词检索 + 向量语义检索 → 各召回 retrieval_k 个
        2. EnsembleRetriever 加权融合
        3. CrossEncoder 重排序 → 精选 rerank_top_n 个最相关文档

    Returns:
        ContextualCompressionRetriever (或降级为 EnsembleRetriever)
    """
    logger.info("构建混合检索器 + 重排序...")
    logger.info(
        f"参数: BM25权重={config.BM25_WEIGHT}, 向量权重={config.VECTOR_WEIGHT}, "
        f"召回={config.RETRIEVAL_K}, 重排序保留={config.RERANK_TOP_N}"
    )

    # 步骤1: BM25 检索器
    doc_texts = [doc.page_content for doc in documents]
    bm25_retriever = BM25Retriever.from_texts(
        doc_texts,
        metadatas=[doc.metadata for doc in documents],  # 保留元数据
    )
    bm25_retriever.k = config.RETRIEVAL_K

    # 步骤2: 向量检索器
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVAL_K}
    )

    # 步骤3: 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[config.BM25_WEIGHT, config.VECTOR_WEIGHT],
    )

    # 步骤4: 重排序
    if not config.RERANK_ENABLED:
        logger.info("重排序已禁用，直接使用混合检索")
        return ensemble_retriever

    try:
        import socket
        import urllib.request

        # 先快速检测网络连通性
        try:
            urllib.request.urlopen("https://huggingface.co", timeout=5)
        except Exception:
            logger.warning("HuggingFace 不可达，跳过重排序模型下载")
            return ensemble_retriever

        cross_encoder = CrossEncoder(
            config.RERANK_MODEL, max_length=config.RERANK_MAX_LENGTH
        )
        reranker = CrossEncoderReranker(
            model=cross_encoder, top_n=config.RERANK_TOP_N
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ensemble_retriever
        )
        logger.info("重排序模型加载成功")
        return compression_retriever

    except Exception as e:
        logger.warning(f"重排序模型加载失败: {e}，降级使用混合检索（无重排序）")
        return ensemble_retriever


# ============================================================================
# 7. 初始化 RAG 系统 (主入口)
# ============================================================================
def init_rag(
    use_ocr: bool = False,
    use_hybrid: bool = True,
) -> Optional[RetrievalQA]:
    """
    初始化 RAG 系统

    Args:
        use_ocr: 是否启用 OCR 处理扫描版 PDF
        use_hybrid: 是否启用混合检索 + 重排序

    Returns:
        RetrievalQA: 问答链对象，失败时返回 None
    """
    global cached_documents

    logger.info("=" * 40)
    logger.info("初始化 RAG 系统...")
    config.ensure_dirs()
    start_time = datetime.now()

    # --- 步骤1: 加载或构建向量数据库 ---
    vectorstore = None
    documents = None

    if check_cache_valid():
        logger.info("检测到有效缓存，直接加载")
        try:
            vectorstore, embeddings = load_cached_vectorstore()

            # 增量更新：检测新增/修改/删除的 PDF
            update_summary = run_incremental_update(vectorstore)
            if any(v > 0 for v in update_summary.values()):
                logger.info(
                    f"增量更新完成: 新增{update_summary['new']}, "
                    f"修改{update_summary['modified']}, "
                    f"删除{update_summary['deleted']}, "
                    f"新切片{update_summary['chunks_added']}"
                )

            # BM25 需要原始文档，从 PDF 重新加载
            if use_hybrid and cached_documents is None:
                logger.info("重新加载文档用于混合检索...")
                all_documents = []
                pdf_files = [
                    f
                    for f in os.listdir(config.PDF_FOLDER)
                    if f.lower().endswith(".pdf")
                ]
                for pdf_file in pdf_files:
                    full_path = os.path.join(config.PDF_FOLDER, pdf_file)
                    try:
                        loader = PyPDFLoader(full_path)
                        all_documents.extend(loader.load())
                    except Exception as e:
                        logger.warning(f"加载 {pdf_file} 失败: {e}")
                cached_documents = all_documents
            documents = cached_documents

        except Exception as e:
            logger.warning(f"加载缓存失败: {e}，将重新构建")
            vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
            if vectorstore is None:
                return None
    else:
        logger.info("未检测到有效缓存，开始构建...")
        vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
        if vectorstore is None:
            return None

    # --- 步骤2: 构建检索器 ---
    if use_hybrid and documents and len(documents) > 0:
        logger.info("启用混合检索 + 重排序模式")
        retriever = create_hybrid_retriever_with_rerank(vectorstore, documents)
    else:
        logger.info("使用标准向量检索模式")
        retriever = vectorstore.as_retriever()

    # --- 步骤3: 配置 LLM ---
    llm = _create_llm()

    # --- 步骤4: Prompt 模板 ---
    template = """根据以下医学文献回答问题。如果上下文无相关信息，回答"未找到相关信息"。

上下文：
{context}

问题：{question}
回答："""

    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # --- 步骤5: 创建问答链 ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
        verbose=False,  # 生产环境关闭 verbose，日志已足够
    )

    init_time = datetime.now() - start_time
    logger.info(f"RAG 初始化完成，总耗时 {init_time.total_seconds():.2f} 秒")

    return qa_chain


# ============================================================================
# 8. 流式 RAG (LCEL 链，支持 token 级别流式输出)
# ============================================================================
def init_rag_streaming(
    use_ocr: bool = False,
    use_hybrid: bool = True,
) -> Optional[Tuple[object, object]]:
    """
    初始化支持流式输出的 RAG 系统

    返回 (retriever, streaming_chain):
        - retriever:  用于预先获取参考文档
        - streaming_chain: LCEL 链，支持 .stream(query) 逐 token 输出

    用法:
        retriever, chain = init_rag_streaming()
        docs = retriever.invoke(query)              # 获取来源文档
        for chunk in chain.stream(query):           # 流式生成
            print(chunk, end="")
    """


def init_retriever(use_ocr: bool = False, use_hybrid: bool = True):
    """
    仅初始化检索器（不包含 LLM 链），供前端动态切换 LLM 使用。

    Returns:
        retriever: 检索器对象
    """
    global cached_documents

    logger.info("=" * 40)
    logger.info("初始化检索器...")
    config.ensure_dirs()
    start_time = datetime.now()

    # --- 加载或构建向量数据库 ---
    vectorstore = None
    documents = None

    if check_cache_valid():
        logger.info("检测到有效缓存，直接加载")
        try:
            vectorstore, embeddings = load_cached_vectorstore()
            update_summary = run_incremental_update(vectorstore)
            if any(v > 0 for v in update_summary.values()):
                logger.info(f"增量更新: 新增{update_summary['new']}, 修改{update_summary['modified']}")

            if use_hybrid and cached_documents is None:
                logger.info("重新加载文档用于混合检索...")
                all_documents = []
                pdf_files = [f for f in os.listdir(config.PDF_FOLDER) if f.lower().endswith(".pdf")]
                for pdf_file in pdf_files:
                    full_path = os.path.join(config.PDF_FOLDER, pdf_file)
                    try:
                        loader = PyPDFLoader(full_path)
                        all_documents.extend(loader.load())
                    except Exception as e:
                        logger.warning(f"加载 {pdf_file} 失败: {e}")
                cached_documents = all_documents
            documents = cached_documents
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}，将重新构建")
            vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
            if vectorstore is None:
                return None
    else:
        logger.info("未检测到有效缓存，开始构建...")
        vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
        if vectorstore is None:
            return None

    # --- 构建检索器 ---
    if use_hybrid and documents and len(documents) > 0:
        logger.info("启用混合检索 + 重排序模式")
        retriever = create_hybrid_retriever_with_rerank(vectorstore, documents)
    else:
        logger.info("使用标准向量检索模式")
        retriever = vectorstore.as_retriever()

    init_time = datetime.now() - start_time
    logger.info(f"检索器初始化完成，总耗时 {init_time.total_seconds():.2f} 秒")

    return retriever


# ============================================================================
# 8. 流式 RAG (LCEL 链)
# ============================================================================
def init_rag_streaming(
    use_ocr: bool = False,
    use_hybrid: bool = True,
) -> Optional[Tuple[object, object]]:
    """
    初始化支持流式输出的 RAG 系统。返回 (retriever, streaming_chain)。
    """
    global cached_documents

    logger.info("=" * 40)
    logger.info("初始化流式 RAG 系统...")
    config.ensure_dirs()
    start_time = datetime.now()

    # --- 步骤1: 加载或构建向量数据库 (与 init_rag 相同) ---
    vectorstore = None
    documents = None

    if check_cache_valid():
        logger.info("检测到有效缓存，直接加载")
        try:
            vectorstore, embeddings = load_cached_vectorstore()

            # 增量更新：检测新增/修改/删除的 PDF
            update_summary = run_incremental_update(vectorstore)
            if any(v > 0 for v in update_summary.values()):
                logger.info(
                    f"增量更新完成: 新增{update_summary['new']}, "
                    f"修改{update_summary['modified']}, "
                    f"删除{update_summary['deleted']}, "
                    f"新切片{update_summary['chunks_added']}"
                )

            if use_hybrid and cached_documents is None:
                logger.info("重新加载文档用于混合检索...")
                all_documents = []
                pdf_files = [
                    f for f in os.listdir(config.PDF_FOLDER)
                    if f.lower().endswith(".pdf")
                ]
                for pdf_file in pdf_files:
                    full_path = os.path.join(config.PDF_FOLDER, pdf_file)
                    try:
                        loader = PyPDFLoader(full_path)
                        all_documents.extend(loader.load())
                    except Exception as e:
                        logger.warning(f"加载 {pdf_file} 失败: {e}")
                cached_documents = all_documents
            documents = cached_documents
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}，将重新构建")
            vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
            if vectorstore is None:
                return None
    else:
        logger.info("未检测到有效缓存，开始构建...")
        vectorstore, embeddings, documents = process_pdfs_and_build_vectorstore(use_ocr)
        if vectorstore is None:
            return None

    # --- 步骤2: 构建检索器 ---
    if use_hybrid and documents and len(documents) > 0:
        logger.info("启用混合检索 + 重排序模式")
        retriever = create_hybrid_retriever_with_rerank(vectorstore, documents)
    else:
        logger.info("使用标准向量检索模式")
        retriever = vectorstore.as_retriever()

    # --- 步骤3: 配置 LLM (streaming) ---
    llm = _create_llm()

    # --- 步骤4: Prompt 模板 ---
    template = """根据以下医学文献回答问题。如果上下文无相关信息，回答"未找到相关信息"。

上下文：
{context}

问题：{question}
回答："""

    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # --- 步骤5: 构建 LCEL 流式链 ---
    def _format_docs(docs: List[Document]) -> str:
        """将检索到的文档格式化为上下文字符串"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            source = os.path.basename(source) if source else "未知"
            # 限制每个片段最多 300 字
            content = doc.page_content[:300]
            parts.append(f"[{i}] {content}")
        return "\n\n".join(parts)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    init_time = datetime.now() - start_time
    logger.info(f"流式 RAG 初始化完成，总耗时 {init_time.total_seconds():.2f} 秒")

    return retriever, rag_chain


# ============================================================================
# 9. CLI 入口 (python rag_engine.py)
# ============================================================================
if __name__ == "__main__":
    config.display()

    # 命令行参数
    use_hybrid = "--no-hybrid" not in sys.argv
    use_ocr = "--ocr" in sys.argv

    if "--clear-cache" in sys.argv:
        clear_cache()
        sys.exit(0)
    if "--cache-info" in sys.argv:
        show_cache_info()
        sys.exit(0)

    show_cache_info()

    if use_ocr:
        print("🔍 OCR 模式已启用")
    if not use_hybrid:
        print("📌 混合检索已禁用")

    # 初始化
    qa_chain = init_rag(use_ocr=use_ocr, use_hybrid=use_hybrid)
    if qa_chain is None:
        print(f"\n💡 提示: 请确保 {config.PDF_FOLDER} 目录下有 PDF 文件")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ 医疗 RAG 助手已就绪！")
    print(f"   - 混合检索: {'✅ 启用' if use_hybrid else '❌ 禁用'}")
    print(f"   - 重排序:   {'✅ 启用' if use_hybrid else '❌ 禁用'}")
    print(f"   - 查询重写: ✅ 启用")
    print("💡 命令: --clear-cache / --cache-info / --no-hybrid / --ocr")
    print("=" * 50)

    # 交互循环
    conversation_history: List[str] = []

    while True:
        try:
            query = input("\n❓ 请输入你的问题: ").strip()
            if query.lower() in ["exit", "quit", "退出"]:
                print("👋 感谢使用，再见！")
                break
            if not query:
                continue

            # 查询重写
            final_query = rewrite_query(query, conversation_history)

            # 检索 + 生成
            print("🤔 思考中...", end="\r")
            result = qa_chain.invoke(final_query)

            # 更新对话历史
            conversation_history.append(f"用户: {query}")
            conversation_history.append(f"AI: {result['result']}")
            if len(conversation_history) > config.HISTORY_MAX_ENTRIES:
                conversation_history = conversation_history[-config.HISTORY_MAX_ENTRIES :]

            print(" " * 20, end="\r")
            print(f"🤖 AI: {result['result']}")

            # 参考来源
            print("\n📚 参考来源:")
            for i, doc in enumerate(result["source_documents"][:5]):
                source_file = os.path.basename(
                    doc.metadata.get("source", "未知文件")
                )
                page_num = doc.metadata.get("page", "?")
                preview = doc.page_content.replace("\n", " ").strip()[:80]
                print(f"  [{i+1}] 📄 {source_file} (第{page_num}页)")
                print(f"      📝 {preview}...")

        except KeyboardInterrupt:
            print("\n\n👋 检测到强制中断，已退出。")
            break
        except Exception as e:
            logger.error(f"运行时错误: {e}", exc_info=True)
            print(f"\n💡 发生错误: {e}")
