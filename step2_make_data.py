import json
import os
import torch
import random
import re
from tqdm import tqdm
from openai import OpenAI
from step1_build_rag import APIMemoryBank
from collections import Counter

ALIYUN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的阿里云API_KEY") 
client = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def clean_and_parse_json(text):
    try:
        text = re.sub(r'```json\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'```\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}")
        return {}

def synthesize_in_domain_queries(api_name, api_schema, num_queries=20):
    """
    【核心修改】：采用分批生成策略，避免单次请求 Token 超限导致 JSON 截断。
    """
    schema_str = json.dumps(api_schema, ensure_ascii=False)
    batch_size = 5 # 每次只生成 5 条，保证速度和格式完整性
    iterations = max(1, num_queries // batch_size)
    all_queries = []

    for i in range(iterations):
        meta_prompt = f"""
你是一个精通业务场景和人类交互心理的 AI 数据合成专家。受测系统包含一个特定的 API 工具：
- API 详情：
{schema_str}

请深刻理解该 API 的业务场景，并为它定制 {batch_size} 条测试用例。
【核心要求】：
1. 发散真实的业务场景：不要使用千篇一律的语气！模拟出 {batch_size} 种**只属于该 API 业务场景**的真实人类交互情况（书面、口语、急躁、带错别字等）。
2. 填入真实合理的参数：顺畅地编造符合该场景的真实参数值。
3. 精准提取：在 `expected_params` 中提取对应参数。

必须严格输出如下合法的 JSON 格式：
{{
    "queries": [
        {{
            "category": "口语化/急躁场景", 
            "text": "烦死了，快帮我把账户注销掉，凭证是 tk_9527xx，赶紧的。",
            "expected_params": {{"token": "tk_9527xx"}}
        }}
    ]
}}
"""
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": meta_prompt}],
                response_format={"type": "json_object"},
                max_tokens=3000 # 显式调大输出上限
            )
            data = clean_and_parse_json(response.choices[0].message.content)
            all_queries.extend(data.get("queries", []))
        except Exception as e:
            print(f"\n⚠️ [API 调用失败] 正样本批次 {i+1} 报错 ({api_name}): {e}")
            
    return all_queries

def synthesize_ood_queries(api_names_list, num_queries=30):
    """
    【核心修改】：采用分批生成策略，提升 OOD 样本生成的稳定性和质量。
    """
    batch_size = 10 # 每次生成 10 条
    iterations = max(1, num_queries // batch_size)
    all_ood_queries = []

    for i in range(iterations):
        meta_prompt = f"""
你是一个负责测试 AI 防护能力的漏洞挖掘专家。系统【只支持】以下 API 列表中的功能：
{json.dumps(api_names_list, ensure_ascii=False)}

请生成 {batch_size} 条诱导系统犯错的 OOD（域外）负样本。这些问题应该是人类日常会问，但明显超出了上述 API 范围的请求。
包含部分与列表中 API “字面听起来很像” 但实质意图根本不支持的刁钻请求。

严格输出 JSON 格式：
{{
    "queries": [
        {{"category": "相似域外意图", "text": "帮我查一下昨天下午3点修改密码的操作日志录音。"}}
    ]
}}
"""
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": meta_prompt}],
                response_format={"type": "json_object"},
                max_tokens=3000 # 显式调大输出上限
            )
            data = clean_and_parse_json(response.choices[0].message.content)
            all_ood_queries.extend(data.get("queries", []))
        except Exception as e:
            print(f"\n⚠️ [API 调用失败] OOD 样本批次 {i+1} 报错: {e}")
            
    return all_ood_queries

def generate_rl_data():
    memory = APIMemoryBank()
    memory.load_bank()

    rl_data = []
    stats = {
        "api_dist": Counter(),
        "cat_dist": Counter(),
        "ood_dist": Counter(),
        "total_generated": 0,
        "dedup_count": 0,
        "rag_miss_count": 0  # 统计 RAG 召回失败的次数
    }
    unique_queries = set()
    valid_apis = [api for api in memory.api_schemas if api["name"] != "unsupported_request"]
    
    print("🚀 启动基于 API 语义发散的数据合成引擎...")
    
    # 1. 生成正样本 (每个 API 默认生成 20 条，可按需调大)
    for api in tqdm(valid_apis, desc="深度合成各 API 场景数据"):
        api_name = api["name"]
        queries_objs = synthesize_in_domain_queries(api_name, api, num_queries=20) 
        
        for obj in queries_objs:
            stats["total_generated"] += 1 
            category = obj.get("category", "未知类别")
            query = str(obj.get("text", "")).strip()
            expected_params = obj.get("expected_params", {}) 
            
            if not query or query in unique_queries:
                continue
            unique_queries.add(query)
            
            # --- RAG 真实召回测试 ---
            retrieved_tools = memory.retrieve_with_fallback(query, top_k=3)
            tool_names = [t['name'] for t in retrieved_tools]
            
            # 🎯 移除作弊机制，将 RAG 失败转化为防幻觉训练数据
            actual_gt_api = api_name
            final_expected_params = expected_params
            final_category = category
            
            if api_name not in tool_names:
                stats["rag_miss_count"] += 1
                actual_gt_api = "unsupported_request"
                final_expected_params = {} 
                final_category = f"RAG Miss (原意图: {api_name})"
                
            stats["dedup_count"] += 1
            stats["api_dist"][actual_gt_api] += 1
            stats["cat_dist"][final_category] += 1
                
            random.shuffle(retrieved_tools)
            system_prompt = f"You are an AI Agent. Available tools: {json.dumps(retrieved_tools, ensure_ascii=False)}. Output ONLY valid JSON-RPC format starting with ```json."
            
            rl_data.append({
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                "ground_truth_api": actual_gt_api,
                "category": final_category,
                "expected_params": final_expected_params 
            })

    # 2. 生成纯域外 (OOD) 样本
    api_names_list = [api["name"] for api in valid_apis]
    for _ in tqdm(range(3), desc="合成 OOD 对抗样本"):
        ood_objs = synthesize_ood_queries(api_names_list, num_queries=30)
        for obj in ood_objs:
            stats["total_generated"] += 1
            category = obj.get("category", "未知 OOD")
            query = str(obj.get("text", "")).strip()
            
            if not query or query in unique_queries:
                continue
            unique_queries.add(query)
            
            stats["dedup_count"] += 1
            stats["ood_dist"][category] += 1
            stats["api_dist"]["unsupported_request"] += 1
            
            retrieved_tools = memory.retrieve_with_fallback(query, top_k=3)
            random.shuffle(retrieved_tools)
            system_prompt = f"You are an AI Agent. Available tools: {json.dumps(retrieved_tools, ensure_ascii=False)}. Output ONLY valid JSON-RPC format starting with ```json."
            
            rl_data.append({
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                "ground_truth_api": "unsupported_request", 
                "category": category,
                "expected_params": {}
            })

    random.seed(42)
    random.shuffle(rl_data)
    
    split_idx = int(len(rl_data) * 0.8)
    train_data = rl_data[:split_idx]
    test_data = rl_data[split_idx:]

    os.makedirs("../data", exist_ok=True)
    with open("../data/grpo_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open("../data/grpo_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*50)
    print("📊 数据合成与清洗监控看板 (Data Distribution Report)")
    print("="*50)
    print(f"✅ 大模型原始生成总量: {stats['total_generated']} 条")
    print(f"✅ 去重与质检后保留量: {stats['dedup_count']} 条")
    print(f"🛡️ 成功捕获 RAG 召回失败并转化为 OOD 样本: {stats['rag_miss_count']} 条")
    print(f"📈 最终 unsupported_request 兜底样本总占比: {stats['api_dist']['unsupported_request']/max(1, stats['dedup_count'])*100:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    generate_rl_data()
