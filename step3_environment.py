import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MCPSimulatorEnv:
    def __init__(self, use_llm_judge=True):
        self.R_FORMAT_SUCCESS = 0.5
        self.R_FORMAT_FAIL = -2.0
        self.R_FINAL_SUCCESS = 2.0
        self.R_ERROR_PENALTY = -2.0
        
        self.api_schemas_dict = {}
        try:
            with open("../vector_db/api_schemas.json", "r", encoding="utf-8") as f:
                schemas = json.load(f)
                for api in schemas:
                    self.api_schemas_dict[api["name"]] = api
            self.valid_api_names = set(self.api_schemas_dict.keys())
        except Exception as e:
            print(f"警告: 无法加载 API Schema: {e}")
            self.valid_api_names = set()

        # 【核心修改】：初始化轻量级本地裁判模型
        self.use_llm_judge = use_llm_judge
        self.judge_model = None
        self.judge_tokenizer = None
        
        if self.use_llm_judge:
            print("⚖️ 正在加载 Qwen2.5-0.5B-Instruct 作为语义裁判模型...")
            try:
                # 获取当前分布式进程的对应 GPU，确保 DDP/FSDP 不会多卡冲突
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
                
                judge_id = "Qwen/Qwen2.5-0.5B-Instruct"
                self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
                self.judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_id,
                    torch_dtype=torch.bfloat16,
                    device_map={"": device} # 锁定在当前进程的显卡上
                )
                self.judge_model.eval()
            except Exception as e:
                print(f"⚠️ 裁判模型加载失败，将降级为字符串比对: {e}")
                self.use_llm_judge = False

    def extract_json(self, text):
        match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "method" in data:
                    return True, data
            except json.JSONDecodeError:
                pass
        return False, {}

    def _judge_semantic_match(self, expected, extracted):
        if not self.use_llm_judge or not self.judge_model:
            # 降级方案：放宽的字符串包含匹配
            return str(expected).lower().strip() in str(extracted).lower().strip()
            
        prompt = f"你是一个严格但不死板的语义校验器。请判断【提取参数】是否正确表达了【预期参数】的核心信息。例如'明天下午3点'和'15:00'是一致的。\n预期参数: {expected}\n提取参数: {extracted}\n如果核心语义一致，请只回答 YES。如果有明显缺失或错误，请只回答 NO。"
        messages = [{"role": "user", "content": prompt}]
        text = self.judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.judge_tokenizer(text, return_tensors="pt").to(self.judge_model.device)
        
        with torch.no_grad():
            outputs = self.judge_model.generate(**inputs, max_new_tokens=5, temperature=0.01, do_sample=False)
        response = self.judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        torch.cuda.empty_cache()
        
        return "YES" in response

    def calculate_reward(self, action_text, gt_api, expected_params=None):
        is_valid, action_json = self.extract_json(action_text)
        if not is_valid: return self.R_FORMAT_FAIL

        method_called = action_json["method"]
        params = action_json.get("params", {})
        reward = self.R_FORMAT_SUCCESS
        
        if method_called not in self.valid_api_names:
            return reward - 2.0 
        
        if method_called == gt_api:
            reward += self.R_FINAL_SUCCESS
        else:
            return reward + self.R_ERROR_PENALTY
        
        # 【核心修改】：通过 Judge 模型计算参数语义一致性得分
        if expected_params:
            for key, expected_val in expected_params.items():
                if key not in params:
                    reward -= 1.0  # 漏掉参数，重罚
                else:
                    is_match = self._judge_semantic_match(expected_val, params[key])
                    if not is_match:
                        reward -= 1.0  # 语义不符，重罚
                    else:
                        reward += 0.5  # 语义一致，加分
        return reward

    def get_observation(self, action_text, gt_api=None, expected_params=None):
        is_valid, action_json = self.extract_json(action_text)
        if not is_valid:
            return False, False, "Observation: Invalid JSON format. Must wrap in ```json."
        
        method = action_json.get("method")
        params = action_json.get("params", {})
        
        if method not in self.valid_api_names:
            return False, False, f"Observation: API '{method}' does not exist."
            
        schema = self.api_schemas_dict.get(method, {})
        required_params = schema.get("parameters", {}).get("required", [])
        for req in required_params:
            if req not in params:
                return False, False, f"Observation: Missing required parameter: '{req}'."
                
        env_success = True 
        task_success = False
        
        if gt_api is not None:
            if method == gt_api:
                if method == "unsupported_request":
                    task_success = True
                else:
                    # 【核心修改】：执行反馈闭环中同样使用语义裁判
                    if expected_params:
                        all_match = True
                        for k, v in expected_params.items():
                            if k not in params or not self._judge_semantic_match(v, params[k]):
                                all_match = False
                                break
                        task_success = all_match
                    else:
                        task_success = True 
            else:
                task_success = False
                
        return env_success, task_success, f"Observation: Executed '{method}' successfully with params: {json.dumps(params, ensure_ascii=False)}."

_ENV = None

def get_env():
    global _ENV
    if _ENV is None:
        _ENV = MCPSimulatorEnv()
    return _ENV

def format_reward_func(completions, **kwargs):
    env = get_env()
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        is_valid, _ = env.extract_json(content)
        rewards.append(env.R_FORMAT_SUCCESS if is_valid else env.R_FORMAT_FAIL)
    return rewards

def correctness_reward_func(prompts, completions, ground_truth_api, **kwargs):
    env = get_env()
    rewards = []
    expected_params_list = kwargs.get("expected_params", [{}] * len(prompts))
    for comp, gt_api, exp_params in zip(completions, ground_truth_api, expected_params_list):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        # 【新增逻辑】：将 Dataset 传过来的字符串安全解析回字典
        if isinstance(exp_params, str):
            try:
                exp_params = json.loads(exp_params)
            except:
                exp_params = {}
        # 防止大模型生成了非字典的异常数据导致后续 .items() 报错
        if not isinstance(exp_params, dict):
            exp_params = {}
            
        total_r = env.calculate_reward(content, gt_api, exp_params) - env.R_FORMAT_SUCCESS
        rewards.append(total_r)
    return rewards