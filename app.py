import streamlit as st
from rag_engine import init_rag_streaming, rewrite_query

# ==========================================
# 1. 核心缓存配置
# ==========================================
@st.cache_resource
def get_rag_components():
    """
    初始化流式 RAG 组件，使用缓存。
    返回 (retriever, streaming_chain)
    """
    result = init_rag_streaming(use_hybrid=True)
    if result is None:
        return None, None
    return result


# ==========================================
# 2. 页面基础配置
# ==========================================
st.set_page_config(page_title="医疗 RAG 助手", layout="wide")

# ==========================================
# 3. 初始化模型 (带加载动画)
# ==========================================
with st.spinner("🚀 正在加载医疗知识库模型（首次运行可能较慢）..."):
    try:
        retriever, rag_chain = get_rag_components()
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

# 检查是否加载成功
if retriever is None or rag_chain is None:
    st.error("⚠️ 模型初始化失败！请确保当前目录下有 PDF 文件。")
    st.stop()

# ==========================================
# 4. 初始化会话状态
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ==========================================
# 5. 页面布局：左右分栏
# ==========================================
col_chat, col_ref = st.columns([3, 1])

with col_chat:
    st.header("🩺 医疗智能问答")

    # 渲染历史聊天记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果消息包含来源，显示折叠的来源信息
            if msg.get("sources"):
                with st.expander(f"📚 查看参考来源 ({len(msg['sources'])} 篇)"):
                    for i, doc in enumerate(msg["sources"], 1):
                        src = (
                            doc.metadata.get("source", "未知文件")
                            .replace("\\", "/")
                            .split("/")[-1]
                        )
                        page = doc.metadata.get("page", "?")
                        st.caption(f"[{i}] {src} (第{page}页)")
                        st.text(doc.page_content[:200] + "...")

    # 清除对话按钮
    if st.session_state.messages:
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.session_state.sources = []
            st.session_state.conversation_history = []
            st.rerun()

# ==========================================
# 6. 聊天输入与处理 (流式)
# ==========================================
if query := st.chat_input("请输入您的医学问题..."):
    # A. 显示用户提问
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # B. 获取 AI 回答 (流式)
    with st.chat_message("assistant"):
        try:
            # 步骤1：查询重写
            history = st.session_state.conversation_history
            rewritten_query = rewrite_query(query, history)
            if rewritten_query != query:
                st.caption(f"🔄 查询优化: _{rewritten_query}_")

            # 步骤2：检索文档（先获取来源，再流式生成）
            with st.spinner("🔍 正在检索医学文献..."):
                docs = retriever.invoke(rewritten_query)

            # 步骤3：流式生成回答
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner("🤔 正在生成回答..."):
                for chunk in rag_chain.stream(rewritten_query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            # 移除光标
            response_placeholder.markdown(full_response)

            # 步骤4：更新对话历史
            history.append(f"用户: {query}")
            history.append(f"AI: {full_response}")
            if len(history) > 8:
                st.session_state.conversation_history = history[-8:]

            # 步骤5：保存到 session_state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": docs,
            })
            st.session_state.sources = docs

        except Exception as e:
            st.error(f"发生错误: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，回答时发生错误。",
            })

# --- 右侧：引用文献侧边栏 ---
with col_ref:
    st.header("📚 引用文献核对")
    st.caption("✅ 系统状态: 就绪 | 流式输出")
    st.divider()

    current_sources = st.session_state.get("sources", [])

    if current_sources:
        for i, doc in enumerate(current_sources, 1):
            source_name = (
                doc.metadata.get("source", "未知文件")
                .replace("\\", "/")
                .split("/")[-1]
            )
            page_num = doc.metadata.get("page", "?")
            content_preview = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )

            with st.expander(f"来源 {i}: {source_name}", expanded=False):
                st.write(f"**第 {page_num} 页**")
                st.info(content_preview)
    else:
        st.info("等待提问以显示引用...")
