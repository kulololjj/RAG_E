import streamlit as st

st.set_page_config(page_title="医疗 RAG 助手", layout="wide")
st.title("🩺 这是一个纯净的测试页面")
st.write("如果你能看到这行字，说明基础页面渲染完全没问题！")

# 故意把模型加载注释掉，看看网页能不能秒开
# from textc1 import init_rag
# with st.spinner("正在加载模型..."):
#     qa_chain = init_rag()