"""
医疗 RAG 系统 - 质量评估模块

支持两种评估模式:
    1. 内置评估 (无需额外依赖) - 上下文命中率、答案覆盖率
    2. RAGAS 评估 (需 pip install ragas) - Faithfulness, Answer Relevancy,
       Context Precision, Context Recall

用法:
    python eval.py                    # 运行完整评估
    python eval.py --simple           # 仅运行内置评估（无额外依赖）
    python eval.py --questions 10     # 指定测试问题数量
"""

import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rag_engine import config, init_rag_streaming, rewrite_query

# ============================================================================
# 1. 医疗测试问题集
# ============================================================================
MEDICAL_TEST_QUESTIONS = [
    # 皮肤科
    "脂溢性皮炎的主要症状有哪些？",
    "脂溢性皮炎的常见治疗方法是什么？",
    "脂溢性皮炎与银屑病如何鉴别诊断？",
    "婴儿脂溢性皮炎和成人有什么不同？",
    "马拉色菌在脂溢性皮炎中的作用是什么？",
    # 如果知识库包含其他领域，可扩展更多问题
    "常用抗真菌药物有哪些种类？",
    "皮炎患者日常护理需要注意什么？",
    "哪些因素会加重脂溢性皮炎的症状？",
    "脂溢性皮炎是否具有遗传性？",
    "头皮屑和脂溢性皮炎有什么关系？",
]


def truncate(text: str, max_len: int = 300) -> str:
    """截断文本"""
    return text if len(text) <= max_len else text[:max_len] + "..."


# ============================================================================
# 2. 内置评估指标 (无需额外依赖)
# ============================================================================
class BuiltinEvaluator:
    """使用简单的 token 重叠和启发式方法评估 RAG 质量"""

    @staticmethod
    def context_hit_rate(
        question: str, contexts: List[str], reference_keywords: List[str]
    ) -> float:
        """
        上下文命中率：检索到的上下文中包含多少参考关键词

        Args:
            question: 用户问题
            contexts: 检索到的文档内容列表
            reference_keywords: 期望出现的关键词

        Returns:
            float: 0~1 之间的命中率
        """
        all_context_text = " ".join(contexts).lower()
        if not reference_keywords:
            return 0.0

        hits = sum(
            1 for kw in reference_keywords if kw.lower() in all_context_text
        )
        return hits / len(reference_keywords)

    @staticmethod
    def answer_coverage(
        answer: str, contexts: List[str], threshold: float = 0.5
    ) -> float:
        """
        答案覆盖率：回答中的内容有多少能在上下文中找到支撑

        使用简单的 unigram 重叠计算
        """
        if not answer or not contexts:
            return 0.0

        answer_tokens = set(answer)
        context_tokens = set(" ".join(contexts))

        if not answer_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    @staticmethod
    def retrieval_redundancy(contexts: List[str]) -> float:
        """
        检索冗余度：检索到的文档之间的重复程度
        值越高说明检索结果越冗余
        """
        if len(contexts) <= 1:
            return 0.0

        redundancies = []
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                tokens_i = set(contexts[i])
                tokens_j = set(contexts[j])
                if not tokens_i or not tokens_j:
                    continue
                overlap = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                redundancies.append(overlap)

        return sum(redundancies) / len(redundancies) if redundancies else 0.0

    @staticmethod
    def response_length_penalty(answer: str, min_len: int = 20) -> float:
        """回答长度惩罚：过短的回答可能质量不佳"""
        length = len(answer)
        if length < min_len:
            return length / min_len
        return 1.0


# ============================================================================
# 3. RAGAS 评估器 (可选)
# ============================================================================
class RAGASEvaluator:
    """使用 RAGAS 框架进行专业评估"""

    def __init__(self):
        self.available = False
        try:
            from langchain_ollama import ChatOllama

            self.eval_llm = ChatOllama(
                model=config.LLM_MODEL,
                base_url=config.LLM_BASE_URL,
                temperature=0,
            )
            self.available = True
        except Exception as e:
            print(f"⚠️ RAGAS 评估器初始化失败: {e}")

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict:
        """运行 RAGAS 评估"""
        if not self.available:
            return {"error": "RAGAS 不可用"}

        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from ragas.dataset_schema import SingleTurnSample

            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truths:
                metrics.append(context_recall)

            results = []
            for i, question in enumerate(questions):
                sample = SingleTurnSample(
                    user_input=question,
                    response=answers[i] if i < len(answers) else "",
                    retrieved_contexts=(
                        contexts_list[i] if i < len(contexts_list) else []
                    ),
                    reference=(
                        ground_truths[i]
                        if ground_truths and i < len(ground_truths)
                        else None
                    ),
                )
                scores = {}
                for metric in metrics:
                    try:
                        score = metric.score(sample)
                        scores[metric.name] = score
                    except Exception as e:
                        scores[metric.name] = f"error: {e}"
                results.append({"question": question, "scores": scores})

            # 计算均值
            avg_scores = {}
            metric_names = [m.name for m in metrics]
            for name in metric_names:
                values = [
                    r["scores"].get(name, 0)
                    for r in results
                    if isinstance(r["scores"].get(name), (int, float))
                ]
                avg_scores[name] = sum(values) / len(values) if values else 0.0

            return {
                "per_question": results,
                "average_scores": avg_scores,
            }

        except ImportError:
            return {"error": "请安装 ragas: pip install ragas"}
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# 4. 评估运行器
# ============================================================================
class EvalRunner:
    """评估运行器：协调检索、生成、评估全流程"""

    def __init__(self, use_hybrid: bool = True):
        self.use_hybrid = use_hybrid
        self.rag_components = None
        self.builtin = BuiltinEvaluator()

    def _ensure_initialized(self):
        """懒初始化 RAG 组件"""
        if self.rag_components is None:
            print("🚀 正在初始化 RAG 系统...")
            result = init_rag_streaming(use_hybrid=self.use_hybrid)
            if result is None:
                raise RuntimeError("RAG 初始化失败")
            self.rag_components = result

    def run_single(
        self, question: str, history: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """运行单次问答，返回 (answer, contexts)"""
        self._ensure_initialized()
        retriever, rag_chain = self.rag_components

        # 查询重写
        rewritten = rewrite_query(question, history or [])

        # 检索
        docs = retriever.invoke(rewritten)
        contexts = [doc.page_content for doc in docs]

        # 生成
        answer = rag_chain.invoke(rewritten)

        return answer, contexts

    def evaluate_builtin(
        self,
        questions: List[str],
        reference_keywords_list: Optional[List[List[str]]] = None,
    ) -> Dict:
        """
        使用内置指标运行评估
        """
        print("\n" + "=" * 60)
        print("📊 内置评估开始")
        print("=" * 60)

        results = []
        total_context_hit = 0.0
        total_coverage = 0.0
        total_redundancy = 0.0
        total_time = 0.0

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 评估: {truncate(question, 60)}")

            start = time.time()
            answer, contexts = self.run_single(question)
            elapsed = time.time() - start
            total_time += elapsed

            # 关键词 (如果未提供或不够，自动从问题中提取)
            if reference_keywords_list and i - 1 < len(reference_keywords_list):
                keywords = reference_keywords_list[i - 1]
            else:
                keywords = self._extract_keywords(question)

            hit_rate = self.builtin.context_hit_rate(question, contexts, keywords)
            coverage = self.builtin.answer_coverage(answer, contexts)
            redundancy = self.builtin.retrieval_redundancy(contexts)
            length_ok = self.builtin.response_length_penalty(answer)

            total_context_hit += hit_rate
            total_coverage += coverage
            total_redundancy += redundancy

            results.append(
                {
                    "question": question,
                    "answer": truncate(answer, 200),
                    "context_count": len(contexts),
                    "context_hit_rate": round(hit_rate, 3),
                    "answer_coverage": round(coverage, 3),
                    "retrieval_redundancy": round(redundancy, 3),
                    "answer_length": len(answer),
                    "time_seconds": round(elapsed, 2),
                }
            )

            print(f"  命中率: {hit_rate:.2%} | 覆盖率: {coverage:.2%} | "
                  f"冗余度: {redundancy:.2%} | 耗时: {elapsed:.1f}s")

        n = len(questions)
        summary = {
            "total_questions": n,
            "avg_context_hit_rate": round(total_context_hit / n, 3),
            "avg_answer_coverage": round(total_coverage / n, 3),
            "avg_retrieval_redundancy": round(total_redundancy / n, 3),
            "avg_time_seconds": round(total_time / n, 2),
            "total_time_seconds": round(total_time, 2),
        }

        return {"results": results, "summary": summary}

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """从问题中提取关键词（简单实现）"""
        # 移除常见疑问词和停用词
        stop_words = {
            "的", "了", "是", "在", "有", "和", "与", "或", "不", "这",
            "那", "什么", "哪些", "如何", "怎么", "为什么", "是否",
            "可以", "需要", "应该", "能够", "会", "吗", "呢", "啊",
        }
        # 简单分词 (中文按字切分不够好，这里简化处理)
        words = []
        for char in question:
            if char not in stop_words and ord(char) > 127:
                words.append(char)
        # 返回前10个有意义的字作为关键词
        return list(set(words[:10])) if words else question.split()


# ============================================================================
# 5. 报告生成
# ============================================================================
def generate_report(
    builtin_result: Dict,
    ragas_result: Optional[Dict] = None,
    output_file: Optional[str] = None,
):
    """生成评估报告"""
    summary = builtin_result.get("summary", {})

    report = []
    report.append("=" * 60)
    report.append("📋 医疗 RAG 系统 - 质量评估报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"LLM 模型: {config.LLM_MODEL}")
    report.append(f"Embedding: {config.EMBEDDING_MODEL}")
    report.append(f"分块大小: {config.CHUNK_SIZE}")
    report.append("=" * 60)

    report.append("\n## 内置指标")
    report.append(f"  测试问题数:        {summary.get('total_questions', 0)}")
    report.append(f"  上下文命中率:      {summary.get('avg_context_hit_rate', 0):.2%}")
    report.append(f"  答案覆盖率:        {summary.get('avg_answer_coverage', 0):.2%}")
    report.append(f"  检索冗余度:        {summary.get('avg_retrieval_redundancy', 0):.2%}")
    report.append(f"  平均响应时间:      {summary.get('avg_time_seconds', 0):.2f}s")
    report.append(f"  总耗时:            {summary.get('total_time_seconds', 0):.2f}s")

    # 评级
    hit = summary.get("avg_context_hit_rate", 0)
    cov = summary.get("avg_answer_coverage", 0)
    if hit > 0.7 and cov > 0.5:
        grade = "A (优秀)"
    elif hit > 0.5 and cov > 0.3:
        grade = "B (良好)"
    elif hit > 0.3:
        grade = "C (一般)"
    else:
        grade = "D (需要改进)"
    report.append(f"\n  综合评级:          {grade}")

    if ragas_result and "average_scores" in ragas_result:
        report.append("\n## RAGAS 指标")
        for name, score in ragas_result["average_scores"].items():
            report.append(f"  {name}: {score:.3f}")

    # 每个问题的详细结果
    report.append("\n## 逐题详情")
    for i, r in enumerate(builtin_result.get("results", []), 1):
        report.append(f"\n  [{i}] {r['question']}")
        report.append(f"      回答: {truncate(r['answer'], 150)}")
        report.append(f"      上下文数: {r['context_count']} | "
                      f"命中率: {r['context_hit_rate']:.2%} | "
                      f"覆盖率: {r['answer_coverage']:.2%}")

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n📁 报告已保存至: {output_file}")

    return report_text


# ============================================================================
# 6. CLI 入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗 RAG 质量评估")
    parser.add_argument(
        "--simple", action="store_true", help="仅使用内置评估（无 RAGAS）"
    )
    parser.add_argument(
        "--questions", type=int, default=5, help="测试问题数量 (默认5)"
    )
    parser.add_argument(
        "--no-hybrid", action="store_true", help="禁用混合检索"
    )
    parser.add_argument(
        "--output", type=str, default="eval_report.txt", help="报告输出文件"
    )
    parser.add_argument(
        "--ragas", action="store_true", help="启用 RAGAS 评估 (需 pip install ragas)"
    )
    args = parser.parse_args()

    # 选择问题子集
    test_questions = MEDICAL_TEST_QUESTIONS[: min(args.questions, len(MEDICAL_TEST_QUESTIONS))]
    print(f"📝 使用 {len(test_questions)} 个测试问题")

    # 可选：定义参考关键词
    reference_keywords = [
        ["脂溢性皮炎", "症状", "红斑", "鳞屑", "瘙痒"],  # Q1
        ["治疗", "药物", "抗真菌", "激素", "护理"],       # Q2
        ["鉴别", "银屑病", "红斑", "鳞屑"],               # Q3
        ["婴儿", "成人", "区别"],                         # Q4
        ["马拉色菌", "真菌", "发病"],                      # Q5
    ][: len(test_questions)]

    # 运行内置评估
    runner = EvalRunner(use_hybrid=not args.no_hybrid)
    builtin_result = runner.evaluate_builtin(test_questions, reference_keywords)

    # 可选：运行 RAGAS 评估
    ragas_result = None
    if args.ragas:
        print("\n🔬 尝试运行 RAGAS 评估...")
        ragas_eval = RAGASEvaluator()
        if ragas_eval.available:
            questions = [r["question"] for r in builtin_result["results"]]
            answers = [r["answer"] for r in builtin_result["results"]]
            contexts = [
                list(r.get("contexts", [])) for r in builtin_result["results"]
            ]
            ragas_result = ragas_eval.evaluate(questions, answers, contexts)

    # 生成报告
    generate_report(builtin_result, ragas_result, output_file=args.output)
