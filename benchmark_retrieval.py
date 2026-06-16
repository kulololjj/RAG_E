"""
检索策略 A/B 对比：纯向量 vs 混合检索+重排序
测量 Top-5 命中率提升
"""
import sys
sys.path.insert(0, ".")

from rag_engine import (
    init_retriever,
    process_pdfs_and_build_vectorstore,
    create_hybrid_retriever_with_rerank,
    load_cached_vectorstore,
    config,
)

# 测试集：问题 + 期望命中的文档关键词
TEST_CASES = [
    {
        "question": "脂溢性皮炎的主要症状有哪些？",
        "expected_keywords": ["红斑", "鳞屑", "瘙痒", "油腻", "炎症"],
    },
    {
        "question": "脂溢性皮炎怎么治疗？",
        "expected_keywords": ["酮康唑", "抗真菌", "激素", "治疗", "药物"],
    },
    {
        "question": "马拉色菌的作用是什么？",
        "expected_keywords": ["真菌", "马拉色菌", "感染", "定植", "免疫"],
    },
    {
        "question": "婴儿脂溢性皮炎有什么特点？",
        "expected_keywords": ["婴儿", "摇篮帽", "结痂", "头皮", "黄色"],
    },
    {
        "question": "如何鉴别脂溢性皮炎和银屑病？",
        "expected_keywords": ["鉴别", "银屑病", "红斑", "鳞屑", "区别"],
    },
    {
        "question": "脂溢性皮炎的发病机制是什么？",
        "expected_keywords": ["皮脂", "免疫", "真菌", "屏障", "机制"],
    },
    {
        "question": "外用抗真菌药物有哪些？",
        "expected_keywords": ["酮康唑", "咪康唑", "抗真菌", "外用药"],
    },
    {
        "question": "头皮屑是什么原因引起的？",
        "expected_keywords": ["头皮屑", "马拉色菌", "脂溢性", "脱屑"],
    },
]


def evaluate_retriever(retriever, name, top_k=5):
    """评估检索器 Top-K 命中率"""
    total_hits = 0
    total_keywords = 0

    for case in TEST_CASES:
        docs = retriever.invoke(case["question"])
        top_docs = docs[:top_k]
        all_text = " ".join(d.page_content.lower() for d in top_docs)

        hits = sum(
            1 for kw in case["expected_keywords"] if kw.lower() in all_text
        )
        total_hits += hits
        total_keywords += len(case["expected_keywords"])

        print(f"  {case['question'][:30]}... → 命中 {hits}/{len(case['expected_keywords'])}")

    hit_rate = total_hits / total_keywords if total_keywords else 0
    print(f"\n{name} Top-{top_k} 命中率: {hit_rate:.1%}")
    return hit_rate


if __name__ == "__main__":
    print("=" * 60)
    print("检索策略 A/B 对比测试")
    print("=" * 60)

    # 初始化
    print("\n初始化检索器...")
    vs, emb, docs = process_pdfs_and_build_vectorstore()

    # A: 纯向量检索
    print("\n--- A: 纯向量检索 ---")
    vector_only = vs.as_retriever(search_kwargs={"k": 5})
    rate_a = evaluate_retriever(vector_only, "纯向量")

    # B: 混合检索 + 重排序
    print("\n--- B: 混合检索 + 重排序 ---")
    hybrid = create_hybrid_retriever_with_rerank(vs, docs)
    rate_b = evaluate_retriever(hybrid, "混合+重排序")

    print("\n" + "=" * 60)
    print(f"纯向量:   {rate_a:.1%}")
    print(f"混合+重排: {rate_b:.1%}")
    if rate_b > rate_a:
        improvement = (rate_b - rate_a) / rate_a * 100
        print(f"提升:     +{improvement:.0f}%")
    print("=" * 60)
