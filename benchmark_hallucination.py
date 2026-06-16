"""
幻觉率 A/B 对比：无门控 vs 置信度门控
测量 LLM 编造事实的比例
"""
import json
import sys
sys.path.insert(0, ".")

from rag_engine import (
    check_relevance,
    init_retriever,
    rewrite_query,
)

# 测试集：问题 + 期望答案中不应出现的内容（知识库以外的问题）
TEST_CASES = [
    # 知识库内问题（应该能答）
    {"question": "脂溢性皮炎的症状有哪些？", "in_kb": True},
    {"question": "马拉色菌和脂溢性皮炎有什么关系？", "in_kb": True},
    {"question": "成人脂溢性皮炎和婴儿有什么区别？", "in_kb": True},
    {"question": "脂溢性皮炎有哪些治疗方法？", "in_kb": True},
    {"question": "脂溢性皮炎的发病机制是什么？", "in_kb": True},
    # 知识库外问题（应该拒答）
    {"question": "糖尿病足的护理要点是什么？", "in_kb": False},
    {"question": "神经外科手术的术后并发症有哪些？", "in_kb": False},
    {"question": "化疗药物顺铂的副作用是什么？", "in_kb": False},
    {"question": "新型冠状病毒的潜伏期有多长？", "in_kb": False},
    {"question": "疫苗接种后可以喝酒吗？", "in_kb": False},
]


def detect_hallucination(answer, docs, in_kb):
    """检测幻觉：知识库外问题答了非拒答内容 = 幻觉"""
    # 知识库内问题，只要不拒绝回答就不算幻觉（假设检索相关）
    if in_kb:
        return False
    # 知识库外问题，正确拒答不算幻觉
    if "未找到相关信息" in answer or "无法可靠回答" in answer:
        return False
    # 知识库外问题却答了内容 = 幻觉
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("幻觉率 A/B 对比测试")
    print("=" * 60)

    from config import config as cfg
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    llm = ChatOpenAI(
        model=cfg.LLM_MODEL,
        api_key=cfg.LLM_API_KEY or "",
        base_url=cfg.LLM_BASE_URL or "https://api.deepseek.com/v1",
        temperature=0.1,
    )

    retriever = init_retriever(use_hybrid=True)

    # ---- A: 无门控 ----
    print("\n--- A: 无置信度门控 ---")
    hallucinations_a = 0

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="根据以下医学文献回答问题。\n\n上下文：{context}\n\n问题：{question}\n回答："
    )

    for i, case in enumerate(TEST_CASES, 1):
        docs = retriever.invoke(case["question"])

        def _fmt(d): return "\n\n".join(f"[{j+1}] {dc.page_content[:300]}" for j, dc in enumerate(d))
        chain = (
            {"context": lambda x: _fmt(docs), "question": RunnableLambda(lambda x: x)}
            | prompt | llm | StrOutputParser()
        )
        answer = chain.invoke(case["question"])
        is_hallucination = detect_hallucination(answer, docs, case["in_kb"])
        if is_hallucination:
            hallucinations_a += 1

        status = "⚠️ 幻觉" if is_hallucination else "✅ 正常"
        if not case["in_kb"]:
            status += " (应拒答)"
        print(f"  [{i}/{len(TEST_CASES)}] {case['question'][:30]}... → {status}")

    rate_a = hallucinations_a / len(TEST_CASES)

    # ---- B: 有门控 ----
    print("\n--- B: 置信度门控 ---")
    hallucinations_b = 0

    for i, case in enumerate(TEST_CASES, 1):
        docs = retriever.invoke(case["question"])
        relevance = check_relevance(case["question"], docs, llm=llm, threshold=0.4)

        if not relevance.get("can_answer", True):
            answer = f"⚠️ 无法可靠回答。置信度: {relevance.get('confidence', 0):.0%}"
        else:
            def _fmt(d): return "\n\n".join(f"[{j+1}] {dc.page_content[:300]}" for j, dc in enumerate(d))
            chain = (
                {"context": lambda x: _fmt(docs), "question": RunnableLambda(lambda x: x)}
                | prompt | llm | StrOutputParser()
            )
            answer = chain.invoke(case["question"])

        is_hallucination = detect_hallucination(answer, docs, case["in_kb"])
        if is_hallucination:
            hallucinations_b += 1

        status = "⚠️ 幻觉" if is_hallucination else "✅ 正常"
        if not case["in_kb"]:
            status += " (应拒答)"
        print(f"  [{i}/{len(TEST_CASES)}] {case['question'][:30]}... → {status}")

    rate_b = hallucinations_b / len(TEST_CASES)

    print("\n" + "=" * 60)
    print(f"无门控幻觉率:   {rate_a:.1%}")
    print(f"有门控幻觉率:   {rate_b:.1%}")
    if rate_b < rate_a:
        reduction = (rate_a - rate_b) / rate_a * 100
        print(f"降低:           -{reduction:.0f}%")
    print("=" * 60)
