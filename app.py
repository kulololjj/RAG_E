"""
医疗 RAG 助手 - Streamlit Web UI
支持侧边栏切换 LLM Provider（Ollama / OpenAI / Anthropic / DeepSeek）
"""
import os

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from rag_engine import init_retriever, rewrite_query

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="医疗 RAG 助手", page_icon="🩺", layout="wide")

# ==========================================
# 2. 侧边栏 —— LLM Provider 切换
# ==========================================
with st.sidebar:
    st.header("⚙️ LLM 配置")

    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai", "anthropic", "deepseek"],
        index=0,
        format_func=lambda x: {
            "ollama": "🦙 Ollama (本地)",
            "openai": "🤖 OpenAI",
            "anthropic": "🧠 Anthropic Claude",
            "deepseek": "🔍 DeepSeek",
        }[x],
        key="provider",
    )

    default_model = {
        "ollama": "qwen3.5:4b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-6",
        "deepseek": "deepseek-chat",
    }
    # 切 provider 时自动更新 model 和 base_url
    if "prev_provider" not in st.session_state:
        st.session_state.prev_provider = provider
    if st.session_state.prev_provider != provider:
        st.session_state.model = default_model.get(provider, "")
        if provider in ("ollama", "deepseek"):
            st.session_state.base_url = {"ollama": "http://localhost:11434", "deepseek": "https://api.deepseek.com/v1"}.get(provider, "")
        else:
            st.session_state.base_url = ""
        st.session_state.prev_provider = provider

    model = st.text_input("模型名", key="model")
    if not model:
        model = default_model.get(provider, "")

    if provider != "ollama":
        api_key = st.text_input(
            "API Key", type="password",
            value=os.getenv("RAG_LLM_API_KEY", ""),
            key="api_key", placeholder="sk-...",
        )
    else:
        api_key = ""

    if provider in ("ollama", "deepseek"):
        base_url = st.text_input("Base URL", key="base_url")
        if not base_url:
            base_url = {"ollama": "http://localhost:11434", "deepseek": "https://api.deepseek.com/v1"}.get(provider, "")
    else:
        base_url = ""

    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temperature")

    if "llm_connected" in st.session_state and st.session_state.llm_connected:
        st.success(f"✅ 已连接: {provider} / {model}")
    else:
        st.info("👆 配置后提问即自动连接")

    st.divider()
    st.caption("本地 Ollama 需先 ollama serve")
    st.caption("云端 API 需有效的 API Key")


# ==========================================
# 3. 缓存 —— 检索器只初始化一次
# ==========================================
@st.cache_resource
def get_retriever():
    """缓存检索器，切换 LLM 不需要重建"""
    return init_retriever(use_hybrid=True)


# ==========================================
# 4. 动态创建 LLM 链（每次提问时根据侧边栏配置创建）
# ==========================================
# ==========================================
# 4. LLM 工厂（复用 build_llm_chain 的 LLM 创建逻辑）
# ==========================================
def _make_llm():
    """根据侧边栏配置创建 LLM 实例（供查询重写用）"""
    provider = st.session_state.get("provider", "ollama")
    model = st.session_state.get("model", "qwen3.5:4b")
    api_key = st.session_state.get("api_key", "")
    base_url = st.session_state.get("base_url", "http://localhost:11434")
    temperature = st.session_state.get("temperature", 0.1)

    if provider == "ollama":
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY", ""), temperature=temperature)
    elif provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""), temperature=temperature)
    elif provider == "deepseek":
        return ChatOpenAI(model=model or "deepseek-chat", api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
                         base_url=base_url.rstrip("/") or "https://api.deepseek.com/v1", temperature=temperature)
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def build_llm_chain(retriever):
    llm = _make_llm()

    template = """根据以下医学文献回答问题。如果上下文无相关信息，回答"未找到相关信息"。

上下文：
{context}

问题：{question}
回答："""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    def _format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:300]
            parts.append(f"[{i}] {content}")
        return "\n\n".join(parts)

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ==========================================
# 5. 初始化
# ==========================================
with st.spinner("🚀 正在加载医疗知识库（首次运行较慢）..."):
    retriever = get_retriever()

if retriever is None:
    st.error("⚠️ 知识库初始化失败！请确保 bing_pdfs/ 目录下有 PDF 文件。")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ==========================================
# 6. 主界面
# ==========================================
col_chat, col_ref = st.columns([3, 1])

with col_chat:
    st.header("🩺 医疗智能问答")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 参考来源"):
                    for i, doc in enumerate(msg["sources"], 1):
                        src = (
                            doc.metadata.get("source", "未知")
                            .replace("\\", "/")
                            .split("/")[-1]
                        )
                        st.caption(f"[{i}] {src} (第{doc.metadata.get('page','?')}页)")
                        st.text(doc.page_content[:200] + "...")

    if st.session_state.messages:
        if st.button("🗑️ 清除对话"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.session_state.conversation_history = []
            st.rerun()

# ==========================================
# 7. 聊天输入
# ==========================================
if query := st.chat_input("请输入您的医学问题..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            history = st.session_state.conversation_history

            # 用侧边栏配置的 LLM 做查询重写
            rewrite_llm = _make_llm()
            rewritten = rewrite_query(query, history, llm=rewrite_llm)
            if rewritten != query:
                st.caption(f"🔄 查询优化: _{rewritten}_")

            with st.spinner("🔍 检索中..."):
                docs = retriever.invoke(rewritten)

            chain = build_llm_chain(retriever)
            st.session_state.llm_connected = True

            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("🤔 生成中..."):
                for chunk in chain.stream(rewritten):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            history.append(f"用户: {query}")
            history.append(f"AI: {full_response}")
            if len(history) > 8:
                st.session_state.conversation_history = history[-8:]

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": docs,
            })
            st.session_state.sources = docs

        except Exception as e:
            import traceback
            st.error(f"发生错误: {e}")
            st.code(traceback.format_exc())
            st.session_state.llm_connected = False
            st.session_state.messages.append({"role": "assistant", "content": "抱歉，发生错误。"})

# ==========================================
# 8. 右侧引用栏
# ==========================================
with col_ref:
    st.header("📚 引用文献")
    current = st.session_state.get("sources", [])
    if current:
        for i, doc in enumerate(current, 1):
            src = (
                doc.metadata.get("source", "未知")
                .replace("\\", "/")
                .split("/")[-1]
            )
            page = doc.metadata.get("page", "?")
            preview = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )
            with st.expander(f"[{i}] {src}", expanded=False):
                st.write(f"**第 {page} 页**")
                st.info(preview)
    else:
        st.info("等待提问以显示引用...")
