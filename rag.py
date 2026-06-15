"""
医疗 RAG 系统 - CLI 入口

用法:
    python rag.py                      # 正常启动（混合检索 + 缓存）
    python rag.py --clear-cache        # 清理缓存
    python rag.py --cache-info         # 查看缓存信息
    python rag.py --no-hybrid          # 禁用混合检索
    python rag.py --ocr                # 启用 OCR 模式
"""

import sys

from rag_engine import (
    clear_cache,
    config,
    init_rag,
    rewrite_query,
    show_cache_info,
)

if __name__ == "__main__":
    config.display()

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
    conversation_history = []

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
                conversation_history = conversation_history[-config.HISTORY_MAX_ENTRIES:]

            print(" " * 20, end="\r")
            print(f"🤖 AI: {result['result']}")

            # 参考来源
            print("\n📚 参考来源:")
            for i, doc in enumerate(result["source_documents"][:5]):
                source_file = (
                    doc.metadata.get("source", "未知文件")
                    .replace("\\", "/")
                    .split("/")[-1]
                )
                page_num = doc.metadata.get("page", "?")
                preview = doc.page_content.replace("\n", " ").strip()[:80]
                print(f"  [{i+1}] 📄 {source_file} (第{page_num}页)")
                print(f"      📝 {preview}...")

        except KeyboardInterrupt:
            print("\n\n👋 检测到强制中断，已退出。")
            break
        except Exception as e:
            print(f"\n💡 发生错误: {e}")
