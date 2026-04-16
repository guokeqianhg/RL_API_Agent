import json
import time
from tqdm import tqdm
from step1_build_rag import APIMemoryBank

def run_large_scale_eval(test_file="../data/grpo_test.json"):
    print("⏳ 正在加载 LangChain + FAISS + BM25 混合检索库 (已内置 Reranker)...")
    memory = APIMemoryBank()
    try:
        memory.load_bank()
    except FileNotFoundError:
        print("❌ 错误: 找不到向量库文件，请先执行 `python step1_build_rag.py`。")
        return

    # 1. 加载大规模测试数据
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到测试集 {test_file}，请先运行 step2_make_data.py。")
        return

    # 2. 初始化核心评估指标
    metrics = {
        "total_queries": len(test_data),
        "in_domain_total": 0,
        "ood_total": 0,
        "hit_at_1": 0,
        "hit_at_3": 0,
        "hit_at_5": 0,
        "mrr_sum": 0.0,            
        "ood_fallback_success": 0  
    }

    print(f"\n🚀 开始大规模 RAG 性能验证 (共 {metrics['total_queries']} 条测试用例)...")
    start_time = time.time()

    # 3. 遍历测试集进行批量检索评测
    for item in tqdm(test_data, desc="检索测试中", unit="条"):
        prompt_data = item.get("prompt", "")
        query = ""
        if isinstance(prompt_data, list):
            for msg in prompt_data:
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        else:
            query = str(prompt_data)
            
        gt_api = item.get("ground_truth_api")

        # 处理 OOD (域外/兜底) 样本
        if gt_api == "unsupported_request":
            metrics["ood_total"] += 1
            # 这里的 retrieve_with_fallback 内部已经自动完成了重排和兜底追加！
            final_apis = memory.retrieve_with_fallback(query, top_k=3)
            final_names = [api["name"] for api in final_apis]
            if "unsupported_request" in final_names:
                metrics["ood_fallback_success"] += 1
            continue

        # 处理正常业务 (In-Domain) 样本
        metrics["in_domain_total"] += 1
        
        # 这里的 retrieve_debug 内部已经自动帮你做完了重排！
        raw_debug = memory.retrieve_debug(query, top_k=5)
        raw_names = [x["schema"]["name"] for x in raw_debug]

        # 核心指标计算
        hit_rank = -1
        if gt_api in raw_names:
            hit_rank = raw_names.index(gt_api) + 1  
            
        if hit_rank == 1:
            metrics["hit_at_1"] += 1
        if 1 <= hit_rank <= 3:
            metrics["hit_at_3"] += 1
        if 1 <= hit_rank <= 5:
            metrics["hit_at_5"] += 1
            
        if hit_rank > 0:
            metrics["mrr_sum"] += (1.0 / hit_rank)

    # 4. 生成专业评估报告
    total_time = time.time() - start_time
    qps = metrics["total_queries"] / total_time if total_time > 0 else 0
    ind = metrics["in_domain_total"]
    ood = metrics["ood_total"]

    print("\n" + "=" * 60)
    print("📊 工业级 RAG 检索大脑评估报告 (Large-Scale Eval)")
    print("=" * 60)
    print(f"⏱️ 测试耗时: {total_time:.2f} 秒 | 吞吐量 (QPS): {qps:.1f} 查询/秒")
    print(f"🗂️ 样本总数: {metrics['total_queries']} 条 (正样本 {ind} 条, 负样本 {ood} 条)")
    
    if ind > 0:
        print("\n[一、 核心召回指标 (In-Domain Metrics)]")
        print(f"🎯 Hit@1 (首位命中率) : {(metrics['hit_at_1'] / ind * 100):.2f}% ({metrics['hit_at_1']}/{ind})")
        print(f"🎯 Hit@3 (前三命中率) : {(metrics['hit_at_3'] / ind * 100):.2f}% ({metrics['hit_at_3']}/{ind})")
        print(f"🎯 Hit@5 (前五命中率) : {(metrics['hit_at_5'] / ind * 100):.2f}% ({metrics['hit_at_5']}/{ind})")
        print(f"📈 MRR (平均倒数排名) : {(metrics['mrr_sum'] / ind):.4f} (越接近1越好，说明答案越靠前)")
    
    if ood > 0:
        print("\n[二、 边界防御指标 (Out-of-Domain Metrics)]")
        print(f"🛡️ 拒答兜底成功率     : {(metrics['ood_fallback_success'] / ood * 100):.2f}% ({metrics['ood_fallback_success']}/{ood})")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_large_scale_eval()
