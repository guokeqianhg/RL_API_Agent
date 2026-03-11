from step1_build_rag import APIMemoryBank


def run_retrieval_test():
    print("⏳ 正在加载 LangChain + FAISS + BM25 混合检索库...")
    memory = APIMemoryBank()

    try:
        memory.load_bank()
    except FileNotFoundError:
        print("❌ 错误: 找不到向量库文件，请先执行 `python step1_build_rag.py`。")
        return

    test_cases = [
        {
            "query": "帮我定一张明天从北京去上海的高铁票，我叫张三",
            "expected": "BuyTrainTicket",
            "domain": "旅游出行"
        },
        {
            "query": "我出门了，把客厅的空调和灯都关掉",
            "expected": "ControlDevice",
            "domain": "智能家居控制"
        },
        {
            "query": "查一下我尾号8888的账户里还有多少钱",
            "expected": "QueryBalance",
            "domain": "财务管理"
        },
        {
            "query": "下周三下午两点我要和研发团队开对齐会",
            "expected": "AddMeeting",
            "domain": "日程安排"
        },
        {
            "query": "帮我看一下尾号8888这个账户最近都有哪些交易",
            "expected": "QueryTradeDetail",
            "domain": "财务管理"
        },
        {
            "query": "给王总发个邮件，说我下午三点到公司",
            "expected": "SendEmail",
            "domain": "邮件通讯"
        },
        {
            "query": "你能帮我写一篇关于火星文明的科幻小说吗？",
            "expected": "unsupported_request",
            "domain": "域外请求"
        },
    ]

    raw_hit_at_1 = 0
    raw_hit_at_3 = 0
    final_hit = 0

    print("\n" + "=" * 78)
    print("🚀 开始进行检索效果验证")
    print("=" * 78)

    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["expected"]
        domain = case["domain"]

        raw_debug = memory.retrieve_debug(query, top_k=3)
        raw_apis = [x["schema"] for x in raw_debug]
        final_apis = memory.retrieve_with_fallback(query, top_k=3)

        raw_names = [api["name"] for api in raw_apis]
        final_names = [api["name"] for api in final_apis]

        if raw_names and raw_names[0] == expected:
            raw_hit_at_1 += 1
        if expected in raw_names:
            raw_hit_at_3 += 1
        if expected in final_names:
            final_hit += 1

        print(f"\n🗣️ [测试 {idx}] 用户输入: \"{query}\"")
        print(f"📌 预期工具: {expected}    |    业务域: {domain}")

        print("🔎 真实检索 Top-3:")
        for rank, item in enumerate(raw_debug, 1):
            api = item["schema"]
            name = api.get("name", "Unknown")
            desc = api.get("description", "").replace("\n", " ")[:90]

            dense_rank = item["dense_rank"] if item["dense_rank"] is not None else "-"
            bm25_rank = item["bm25_rank"] if item["bm25_rank"] is not None else "-"

            print(f"   [{rank}] {name}")
            print(
                f"       => final={item['final_score']:.6f}, "
                f"dense_score={item['dense_score']:.4f}, "
                f"bm25_score={item['bm25_score']:.4f}, "
                f"dense_rank={dense_rank}, "
                f"bm25_rank={bm25_rank}"
            )
            print(f"       => {desc}...")

        print("🛡️ 下游候选列表（retrieve_with_fallback）:")
        for rank, api in enumerate(final_apis, 1):
            name = api.get("name", "Unknown")
            desc = api.get("description", "").replace("\n", " ")[:90]
            print(f"   [{rank}] {name}")
            print(f"       => {desc}...")

        print(f"✅ Raw Hit@1: {'是' if (raw_names and raw_names[0] == expected) else '否'}")
        print(f"✅ Raw Hit@3: {'是' if expected in raw_names else '否'}")
        print(f"✅ Final Candidate Hit: {'是' if expected in final_names else '否'}")
        print("-" * 78)

    total = len(test_cases)
    print("\n" + "=" * 78)
    print("📊 检索评估汇总")
    print("=" * 78)
    print(f"Raw Hit@1           : {raw_hit_at_1}/{total} = {raw_hit_at_1 / total:.2%}")
    print(f"Raw Hit@3           : {raw_hit_at_3}/{total} = {raw_hit_at_3 / total:.2%}")
    print(f"Final Candidate Hit : {final_hit}/{total} = {final_hit / total:.2%}")
    print("=" * 78)


if __name__ == "__main__":
    run_retrieval_test()