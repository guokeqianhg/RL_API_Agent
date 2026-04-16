import os
# 设置 Hugging Face 国内镜像，防止下载模型时网络中断
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import ast
import json
import pickle
import re
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

import pandas as pd
from rank_bm25 import BM25Okapi

try:
    import jieba
except ImportError:
    jieba = None

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class APIMemoryBank:
    """
    基于 LangChain + FAISS + BM25 + RRF + CrossEncoder 重排的 API 检索库
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        retrieval_k: int = 40,
        candidate_pool_k: int = 5,
        rrf_k: int = 15,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
    ):
        self.base_dir = Path(__file__).resolve().parent

        self.vector_dir = (self.base_dir / "../vector_db").resolve()
        self.default_csv_path = (self.base_dir / "../data/raw_api_bank/all_apis.csv").resolve()

        self.schemas_path = self.vector_dir / "api_schemas.json"
        self.docs_path = self.vector_dir / "retrieval_docs.json"
        self.bm25_path = self.vector_dir / "bm25_corpus.pkl"
        self.faiss_dir = self.vector_dir / "faiss_index"

        self.retrieval_k = retrieval_k
        self.candidate_pool_k = candidate_pool_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.api_schemas = []
        self.documents = []
        self.name2schema = {}
        self.doc_name_order = []
        self.tokenized_corpus = []

        self.vectorstore = None
        self.bm25 = None

        print(f"Loading embedding model {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # ================= [新增：初始化 Cross-Encoder 重排模型] =================
        print("🧠 Loading Reranker model (BAAI/bge-reranker-v2-m3)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=device)
        except Exception as e:
            print(f"⚠️ Reranker 加载失败: {e}。将降级使用基础双路召回。")
            self.reranker = None
        # =====================================================================

    # ----------------------------
    # 基础工具
    # ----------------------------
    def _resolve_path(self, path_like, default_path: Path) -> Path:
        if path_like is None:
            return default_path
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def _read_csv_with_fallback_encodings(self, csv_path: Path) -> pd.DataFrame:
        last_error = None
        for enc in ["utf-8", "gbk", "utf-8-sig"]:
            try:
                return pd.read_csv(csv_path, encoding=enc)
            except UnicodeDecodeError as e:
                last_error = e
        raise last_error

    def _clean_api_info_text(self, api_info_str: str) -> str:
        text = str(api_info_str).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace('""', '"')
        return text

    def _extract_assignment_block(self, text: str, key: str) -> str:
        pattern = rf"{key}\s*=\s*(.*?)(?=\n(?:description|input_parameters|output_parameters)\s*=|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    def _safe_literal_eval(self, expr: str, default):
        if not expr:
            return default
        try:
            return ast.literal_eval(expr)
        except Exception:
            return default

    def _parse_api_info(self, api_info_str: str, api_name: str):
        text = self._clean_api_info_text(api_info_str)

        desc_expr = self._extract_assignment_block(text, "description")
        input_expr = self._extract_assignment_block(text, "input_parameters")
        output_expr = self._extract_assignment_block(text, "output_parameters")

        def _clean_expr(expr):
            if not expr: return expr
            expr = re.sub(r'\btrue\b', 'True', expr)
            expr = re.sub(r'\bfalse\b', 'False', expr)
            expr = re.sub(r'\bnull\b', 'None', expr)
            return expr

        description = self._safe_literal_eval(_clean_expr(desc_expr), api_name)
        if not isinstance(description, str) or not description.strip():
            description = api_name

        input_parameters = self._safe_literal_eval(_clean_expr(input_expr), {})
        if input_parameters in (None, {None}) or not isinstance(input_parameters, dict):
            input_parameters = {}

        output_parameters = self._safe_literal_eval(_clean_expr(output_expr), {})
        if output_parameters in (None, {None}) or not isinstance(output_parameters, dict):
            output_parameters = {}

        return description, input_parameters, output_parameters

    def _split_top_level_commas(self, text: str):
        parts = []
        current = []
        depth = 0
        for ch in str(text):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)

            if ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _parse_signature(self, signature_str: str):
        signature = str(signature_str).strip()
        if not signature or signature.lower() == "nan":
            return []

        if signature.startswith("(") and signature.endswith(")"):
            signature = signature[1:-1]

        parsed = []
        for part in self._split_top_level_commas(signature):
            if ":" in part:
                name, raw_type = part.split(":", 1)
            else:
                name, raw_type = part, "string"

            name = name.strip()
            raw_type = raw_type.strip()
            if name and name != "None":
                parsed.append({"name": name, "raw_type": raw_type or "string"})
        return parsed

    def _parse_cn_param_hints(self, param_text: str):
        text = str(param_text).strip()
        if not text or text.lower() == "nan":
            return []

        chunks = [c.strip() for c in re.split(r"[，,；;]+", text) if c.strip()]
        hints = []
        for chunk in chunks:
            match = re.match(r"(.+?)\s*\((.+?)\)", chunk)
            if match:
                hints.append({
                    "display_name": match.group(1).strip(),
                    "raw_type": match.group(2).strip()
                })
            else:
                hints.append({
                    "display_name": chunk,
                    "raw_type": "string"
                })
        return hints

    def _map_to_json_schema_type(self, raw_type: str) -> str:
        raw_type = str(raw_type).lower().strip()
        if any(x in raw_type for x in ["int", "float", "double", "number"]):
            return "number"
        if any(x in raw_type for x in ["list", "array", "tuple", "set", "["]):
            return "array"
        if any(x in raw_type for x in ["dict", "json", "object", "map"]):
            return "object"
        if "bool" in raw_type:
            return "boolean"
        return "string"

    def _merge_parameter_schema(self, signature_params, cn_hints, parsed_input_params):
        properties = {}
        parsed_required = []

        for param_name, param_details in parsed_input_params.items():
            if param_name in (None, "None"):
                continue

            if isinstance(param_details, dict):
                raw_type = param_details.get("type", "string")
                desc = str(param_details.get("description", "")).strip()
                required = bool(param_details.get("required", True))
            else:
                raw_type = "string"
                desc = str(param_details).strip() if param_details is not None else ""
                required = True

            properties[str(param_name)] = {
                "type": self._map_to_json_schema_type(raw_type),
                "description": desc
            }
            if required:
                parsed_required.append(str(param_name))

        if signature_params:
            required_params = [x["name"] for x in signature_params]
            for i, sig in enumerate(signature_params):
                name = sig["name"]
                if name not in properties:
                    hint = cn_hints[i] if i < len(cn_hints) else {}
                    desc_hint = hint.get("display_name", name)
                    type_hint = hint.get("raw_type", sig.get("raw_type", "string"))
                    properties[name] = {
                        "type": self._map_to_json_schema_type(type_hint),
                        "description": f"参数：{desc_hint}" if desc_hint else ""
                    }
                elif not properties[name].get("description"):
                    hint = cn_hints[i] if i < len(cn_hints) else {}
                    desc_hint = hint.get("display_name", "")
                    if desc_hint:
                        properties[name]["description"] = f"参数：{desc_hint}"
        else:
            required_params = list(parsed_required)

        required_params = [x for x in required_params if x in properties]
        return properties, required_params

    def _build_output_desc(self, output_parameters: dict) -> str:
        if not output_parameters:
            return ""

        chunks = []
        for key, value in output_parameters.items():
            if isinstance(value, dict):
                out_type = value.get("type", "str")
                out_desc = str(value.get("description", "")).strip()
            else:
                out_type = "str"
                out_desc = str(value).strip()

            item = f"{key} ({out_type})"
            if out_desc:
                item += f": {out_desc}"
            chunks.append(item)

        return f" [Returns: {'; '.join(chunks)}]"

    def _split_camel_case(self, text: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", str(text)).strip().lower()

    def _get_operation_phrases(self, api_name: str):
        mapping = {
            "Add": ["添加", "新增", "创建", "安排", "设一个"],
            "Create": ["创建", "新建", "生成"],
            "Delete": ["删除", "移除", "删掉"],
            "Remove": ["删除", "移除", "去掉"],
            "Modify": ["修改", "更改", "调整", "改一下"],
            "Update": ["更新", "修改", "调整"],
            "Query": ["查询", "查看", "查一下", "看一下"],
            "Search": ["搜索", "查找", "搜一下"],
            "Book": ["预订", "预定", "预约"],
            "Buy": ["购买", "买", "订"],
            "Cancel": ["取消", "撤销", "退掉"],
            "Send": ["发送", "发", "通知"],
            "Receive": ["接收", "收取", "获取"],
            "Control": ["控制", "操作", "打开", "关闭"],
            "Get": ["获取", "查看", "看看"],
            "Open": ["开通", "开户", "打开"],
            "Close": ["关闭", "停用"],
            "Record": ["记录", "保存"],
        }
        for prefix, phrases in mapping.items():
            if str(api_name).startswith(prefix):
                return phrases
        return ["处理", "执行"]

    def _get_concept_phrases(self, api_name: str, zh_name: str, scenario: str, description: str):
        text = " ".join([
            str(api_name).lower(),
            str(zh_name).lower(),
            str(scenario).lower(),
            str(description).lower(),
            self._split_camel_case(api_name).lower()
        ])

        concept_map = {
            "meeting": ["会议", "开会", "对齐会", "约个会"],
            "agenda": ["日程", "安排", "行程"],
            "reminder": ["提醒", "提示", "待办提醒"],
            "alarm": ["闹钟", "计时器", "定时提醒"],
            "memo": ["备忘", "备忘录", "记事本"],
            "conflict": ["冲突", "时间重合", "撞期"],
            "device": ["设备", "家电", "灯", "空调", "扫地机器人"],
            "scene": ["场景", "模式", "智能家居场景"],
            "switch": ["开关", "定时开关", "定时任务"],
            "smart": ["智能家居", "全屋智能"],
            "balance": ["余额", "剩余金额", "还有多少钱"],
            "trade": ["交易", "流水", "账单", "明细"],
            "account": ["账户", "银行卡", "账户信息", "开户"],
            "transfer": ["转账", "汇款", "打钱"],
            "exchange": ["汇率", "外汇", "兑换"],
            "stock": ["股票", "股市", "股价"],
            "email": ["邮件", "邮箱", "发信"],
            "message": ["消息", "短信", "发信息"],
            "im": ["即时消息", "聊天消息", "发微信", "发钉钉"],
            "ticket": ["票", "门票", "预约票", "演唱会门票", "景点门票"],
            "train": ["高铁", "火车", "车票", "动车", "买票", "改签"],
            "hotel": ["酒店", "宾馆", "住宿", "订房", "退房"],
            "weather": ["天气", "气温", "下雨", "天气预报"],
            "navigation": ["导航", "路线", "怎么去", "查路线", "公交"],
            "place": ["地点", "位置", "附近", "周边", "哪里有", "商场"],
            "health": ["健康数据", "健康记录", "体征"],
            "symptom": ["症状", "头痛", "发烧", "怎么回事", "哪里不舒服", "疾病检索"],
            "emergency": ["急救", "施救", "受伤了怎么办", "急救知识"],
            "appointment": ["挂号", "看病", "预约医生", "门诊", "取消挂号"],
            "registration": ["挂号信息", "就诊时间", "医生号"],
            "job": ["工作", "岗位", "找工作", "招聘", "职位", "打工", "面试"],
            "company": ["公司", "企业", "年报", "经营范围", "财报", "查公司"],
            "book": ["图书", "书籍", "看书", "找书", "小说"],
            "course": ["课程", "网课", "在线课程", "讲师", "上课"],
            "exam": ["考试", "考试时间", "期末考", "四六级"],
            "conference": ["学术会议", "人工智能会议", "顶会", "AI会议"],
            "paper": ["论文", "paper", "文献", "科研"],
            "express": ["快递", "物流", "查单号", "包裹", "顺丰"],
            "movie": ["电影", "音乐", "找电影", "听歌", "找歌"],
            "song": ["歌曲", "听歌", "识别歌曲", "这是什么歌"],
            "history": ["历史上的今天", "历史事件", "今天发生了什么"],
            "summary": ["摘要", "总结", "概括", "太长不看"],
            "document": ["文档", "文件", "阅读文档", "文档问答"],
            "calculator": ["计算器", "算一下", "等于多少", "数学题"],
            "translate": ["翻译", "英文", "怎么说", "中文"],
            "wiki": ["维基百科", "百科", "解释一下"],
            "dictionary": ["词典", "字典", "查单词", "什么意思"],
            "knowledge": ["知识图谱", "图谱"],
            "caption": ["图像描述", "看图说话", "图里有什么"],
            "speech": ["语音识别", "语音生成", "转文字", "转语音", "念出来"],
            "video": ["视频描述", "看视频", "视频里有什么"],
            "password": ["密码", "口令", "忘记密码", "改密码"],
            "token": ["凭证", "令牌"],
        }

        matched = []
        for key, phrases in concept_map.items():
            if key in text:
                matched.extend(phrases)

        scenario_text = str(scenario)
        if "智能家居" in scenario_text:
            matched.extend(["开灯", "关灯", "开空调", "关空调", "控制家里的设备"])
        if "日程" in scenario_text or "安排" in scenario_text:
            matched.extend(["安排一下", "加个安排", "定个时间"])
        if "财务" in scenario_text:
            matched.extend(["查账户", "查金额", "资金情况"])
        if "旅游" in scenario_text or "出行" in scenario_text:
            matched.extend(["订票", "出行安排", "酒店车票"])
        if "邮件" in scenario_text or "通讯" in scenario_text:
            matched.extend(["发个邮件", "发封邮件", "邮件通知"])

        if zh_name and str(zh_name).lower() != "nan":
            matched.append(str(zh_name).strip())

        dedup = []
        seen = set()
        for x in matched:
            x = str(x).strip()
            if x and x not in seen:
                seen.add(x)
                dedup.append(x)
        return dedup

    def _build_main_retrieval_text(
        self,
        category: str,
        scenario: str,
        zh_name: str,
        api_name: str,
        signature_str: str,
        param_text: str,
        expressions: str,
        description: str,
        properties: dict,
        output_parameters: dict,
    ) -> str:
        param_desc = " ".join(
            [f"{k} {v.get('description', '')}" for k, v in properties.items()]
        ).strip()
        output_names = " ".join(output_parameters.keys()) if output_parameters else "无输出"
        camel_name = self._split_camel_case(api_name)
        operation_phrases = " ".join(self._get_operation_phrases(api_name))
        concept_phrases = " ".join(self._get_concept_phrases(api_name, zh_name, scenario, description))

        return (
            f"类型: {category}\n"
            f"场景: {scenario}\n"
            f"API中文名: {zh_name}\n"
            f"API类名: {api_name}\n"
            f"API类名拆分: {camel_name}\n"
            f"操作语义: {operation_phrases}\n"
            f"概念语义: {concept_phrases}\n"
            f"描述: {description}\n"
            f"参数签名: {signature_str}\n"
            f"参数说明: {param_text}\n"
            f"参数细节: {param_desc}\n"
            f"表达示例: {expressions}\n"
            f"输出字段: {output_names}\n"
            f"用途总结: 该工具适用于 {scenario} 场景下与 {zh_name or api_name} 相关的请求。"
        )

    def _generate_query_style_texts(
        self,
        scenario: str,
        zh_name: str,
        api_name: str,
        description: str,
        param_text: str,
        expressions: str,
    ):
        op_phrases = self._get_operation_phrases(api_name)
        concept_phrases = self._get_concept_phrases(api_name, zh_name, scenario, description)

        texts = []

        for op in op_phrases[:3]:
            for obj in concept_phrases[:4]:
                texts.append(f"用户可能会说：帮我{op}{obj}")
                texts.append(f"用户可能会说：我想{op}{obj}")
                texts.append(f"用户可能会说：请{op}{obj}")

        if str(api_name).startswith("Query") or str(api_name).startswith("Search") or str(api_name).startswith("Get"):
            for obj in concept_phrases[:4]:
                texts.append(f"用户可能会说：查一下{obj}")
                texts.append(f"用户可能会说：帮我看下{obj}")
                texts.append(f"用户可能会说：我想查询{obj}")

        if str(api_name).startswith(("Add", "Book", "Buy", "Create")):
            for obj in concept_phrases[:4]:
                texts.append(f"用户可能会说：帮我安排{obj}")
                texts.append(f"用户可能会说：帮我预定{obj}")
                texts.append(f"用户可能会说：我想新增{obj}")

        if str(api_name).startswith(("Modify", "Update")):
            for obj in concept_phrases[:4]:
                texts.append(f"用户可能会说：帮我修改{obj}")
                texts.append(f"用户可能会说：帮我调整{obj}")

        if str(api_name).startswith(("Delete", "Remove", "Cancel")):
            for obj in concept_phrases[:4]:
                texts.append(f"用户可能会说：帮我取消{obj}")
                texts.append(f"用户可能会说：帮我删除{obj}")

        if scenario and str(scenario).lower() != "nan":
            texts.append(f"用户可能会说：我有一个和{scenario}相关的请求")
            texts.append(f"用户可能会说：帮我处理{scenario}这件事")

        if param_text and str(param_text).lower() != "nan":
            texts.append(f"相关参数包括：{param_text}")
        if expressions and str(expressions).lower() != "nan":
            texts.append(f"相关表达形式：{expressions}")

        concept_blob = " ".join(concept_phrases)
        if "余额" in concept_blob:
            texts.extend([
                "用户可能会说：查一下账户里还有多少钱",
                "用户可能会说：看看余额还剩多少",
            ])
        if "交易" in concept_blob or "流水" in concept_blob:
            texts.extend([
                "用户可能会说：帮我看一下最近都有哪些交易",
                "用户可能会说：查一下账户流水",
            ])
        if "设备" in concept_blob or "灯" in concept_blob or "空调" in concept_blob:
            texts.extend([
                "用户可能会说：把客厅的空调和灯都关掉",
                "用户可能会说：帮我打开家里的设备",
            ])
        if "会议" in concept_blob or "开会" in concept_blob:
            texts.extend([
                "用户可能会说：下周三下午两点安排个会",
                "用户可能会说：帮我拉个对齐会",
            ])
        if "高铁" in concept_blob or "火车" in concept_blob or "车票" in concept_blob:
            texts.extend([
                "用户可能会说：帮我订一张明天从北京去上海的高铁票",
                "用户可能会说：帮我改签我的高铁票",
            ])
        if "邮件" in concept_blob or "邮箱" in concept_blob:
            texts.extend([
                "用户可能会说：给某人发个邮件",
                "用户可能会说：帮我发封邮件通知一下",
            ])

        dedup = []
        seen = set()
        for text in texts:
            t = str(text).strip()
            if not t or t in seen:
                continue
            seen.add(t)
            dedup.append(t)

        return dedup[:8]

    def _tokenize(self, text: str):
        text = str(text).strip()
        if not text:
            return []
        if jieba is not None:
            return jieba.lcut(text)
        return re.findall(r"[A-Za-z0-9_]+|[一-鿿]|[^\s]", text)

    def _build_indices(self):
        self.name2schema = {x["name"]: x for x in self.api_schemas}
        self.doc_name_order = [doc.metadata["name"] for doc in self.documents]
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]

        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _vector_search_scores(self, query: str):
        dense_scores = {}
        dense_rank_map = {}

        if self.vectorstore is None:
            return dense_scores, dense_rank_map

        doc_k = min(max(self.retrieval_k * 6, 40), max(len(self.documents), 1))
        results = self.vectorstore.similarity_search_with_score(query, k=doc_k)

        for rank, (doc, score) in enumerate(results, start=1):
            name = doc.metadata.get("name")
            if not name:
                continue
            sim_score = 1.0 / (1.0 + float(score))
            if name not in dense_scores or rank < dense_rank_map[name]:
                dense_scores[name] = sim_score
                dense_rank_map[name] = rank

        return dense_scores, dense_rank_map

    def _bm25_scores(self, query: str):
        if self.bm25 is None:
            return {}, {}

        tokenized_query = self._tokenize(query)
        raw_scores = self.bm25.get_scores(tokenized_query)

        best_score_by_api = {}
        for idx, score in enumerate(raw_scores):
            name = self.doc_name_order[idx]
            score = float(score)
            if name not in best_score_by_api or score > best_score_by_api[name]:
                best_score_by_api[name] = score

        sorted_names = sorted(best_score_by_api.keys(), key=lambda x: best_score_by_api[x], reverse=True)
        sparse_rank_map = {name: rank for rank, name in enumerate(sorted_names, start=1)}
        return best_score_by_api, sparse_rank_map

    def _hybrid_rank(self, query: str, top_k: int = 3):
        if self.vectorstore is None or self.bm25 is None:
            raise RuntimeError("RAG bank is empty. Please call build_real_api_bank() or load_bank() first.")

        dense_scores, dense_rank_map = self._vector_search_scores(query)
        sparse_scores, sparse_rank_map = self._bm25_scores(query)

        dense_candidates = set(dense_rank_map.keys())
        sparse_candidates = set(list(sorted(sparse_rank_map.keys(), key=lambda x: sparse_rank_map[x]))[: self.retrieval_k * 2])
        candidate_names = dense_candidates | sparse_candidates

        if not candidate_names:
            return []

        ranked = []
        for name in candidate_names:
            dense_rrf = 0.0
            sparse_rrf = 0.0

            if name in dense_rank_map:
                dense_rrf = 1.0 / (self.rrf_k + dense_rank_map[name])
            if name in sparse_rank_map:
                sparse_rrf = 1.0 / (self.rrf_k + sparse_rank_map[name])

            final_score = self.dense_weight * dense_rrf + self.sparse_weight * sparse_rrf
            schema = self.name2schema.get(name)
            if schema is None:
                continue

            ranked.append({
                "name": name,
                "schema": schema,
                "final_score": float(final_score),
                "dense_score": float(dense_scores.get(name, 0.0)),
                "bm25_score": float(sparse_scores.get(name, 0.0)),
                "dense_rank": dense_rank_map.get(name, None),
                "bm25_rank": sparse_rank_map.get(name, None),
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked[:max(top_k, self.candidate_pool_k, self.retrieval_k)]

    # ----------------------------
    # 对外接口
    # ----------------------------
    def build_real_api_bank(self, csv_path=None):
        csv_path = self._resolve_path(csv_path, self.default_csv_path)
        print(f"Parsing real API dataset from {csv_path}...")

        df = self._read_csv_with_fallback_encodings(csv_path)

        self.api_schemas = []
        self.documents = []

        parse_warning_count = 0
        schema_repair_count = 0

        for idx, row in df.iterrows():
            category = str(row.get("类型", "")).strip()
            scenario = str(row.get("应用场景", "")).strip()
            zh_name = str(row.get("API名称", "")).strip()
            api_name = str(row.get("类名", "")).strip()
            signature_str = str(row.get("input_parameters", "")).strip()
            param_text = str(row.get("参数", "")).strip()
            expressions = str(row.get("expressions", "")).strip()
            api_info_str = str(row.get("api_info", "")).strip()

            try:
                description, parsed_input_params, output_params = self._parse_api_info(api_info_str, api_name)
            except Exception as e:
                parse_warning_count += 1
                print(f"⚠️ 第 {idx} 行解析警告: {api_name} 解析失败，将使用兜底信息。错误: {e}")
                description, parsed_input_params, output_params = api_name, {}, {}

            signature_params = self._parse_signature(signature_str)
            cn_hints = self._parse_cn_param_hints(param_text)
            properties, required_params = self._merge_parameter_schema(
                signature_params=signature_params,
                cn_hints=cn_hints,
                parsed_input_params=parsed_input_params,
            )

            if signature_params:
                sig_names = [x["name"] for x in signature_params]
                parsed_names = list(parsed_input_params.keys()) if isinstance(parsed_input_params, dict) else []
                if sig_names != parsed_names:
                    schema_repair_count += 1

            output_desc = self._build_output_desc(output_params)

            schema = {
                "name": api_name,
                "description": f"{description[:350]}{output_desc}",
                "parameters": {
                    "type": "object",
                    "required": required_params,
                    "properties": properties
                }
            }
            self.api_schemas.append(schema)

            main_text = self._build_main_retrieval_text(
                category=category,
                scenario=scenario,
                zh_name=zh_name,
                api_name=api_name,
                signature_str=signature_str,
                param_text=param_text,
                expressions=expressions,
                description=description,
                properties=properties,
                output_parameters=output_params,
            )
            self.documents.append(
                Document(
                    page_content=main_text,
                    metadata={
                        "name": api_name,
                        "doc_type": "main",
                        "scenario": scenario,
                        "zh_name": zh_name,
                        "type": category,
                    }
                )
            )

            query_texts = self._generate_query_style_texts(
                scenario=scenario,
                zh_name=zh_name,
                api_name=api_name,
                description=description,
                param_text=param_text,
                expressions=expressions,
            )
            for qtext in query_texts:
                self.documents.append(
                    Document(
                        page_content=qtext,
                        metadata={
                            "name": api_name,
                            "doc_type": "query_style",
                            "scenario": scenario,
                            "zh_name": zh_name,
                            "type": category,
                        }
                    )
                )

        fallback_schema = {
            "name": "unsupported_request",
            "description": "当可用工具无法满足需求，或者用户请求超出当前业务系统支持范围时，调用该接口以拒绝执行，避免幻觉式调用。",
            "parameters": {
                "type": "object",
                "required": [],
                "properties": {
                    "reason": {"type": "string", "description": "拒绝执行的原因说明"}
                }
            }
        }
        self.api_schemas.append(fallback_schema)

        fallback_texts = [
            "类型: fallback\n场景: 域外请求、闲聊、创作任务、系统不支持的能力\nAPI中文名: 拒绝执行\nAPI类名: unsupported_request\n描述: 当请求超出当前 API 系统支持范围时，拒绝执行。",
            "用户可能会说：帮我写一篇小说",
            "用户可能会说：陪我聊天",
            "用户可能会说：帮我写个故事",
            "用户可能会说：做一个系统外的任务",
            "用户可能会说：这不是你支持的业务范围",
            "用户可能会说：帮我创作内容",
        ]
        for i, text in enumerate(fallback_texts):
            self.documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "name": "unsupported_request",
                        "doc_type": "main" if i == 0 else "query_style",
                        "scenario": "fallback",
                        "zh_name": "拒绝执行",
                        "type": "fallback",
                    }
                )
            )

        print("Building FAISS + BM25 hybrid index...")
        self._build_indices()

        self.vector_dir.mkdir(parents=True, exist_ok=True)

        with open(self.schemas_path, "w", encoding="utf-8") as f:
            json.dump(self.api_schemas, f, ensure_ascii=False, indent=2)

        serializable_docs = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in self.documents
        ]
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        self.vectorstore.save_local(str(self.faiss_dir))

        print(
            f"✅ RAG Database Built! 共 {len(self.api_schemas)} 个工具"
            f" | 检索文档数: {len(self.documents)}"
            f" | 解析告警: {parse_warning_count}"
            f" | Schema 自动修复: {schema_repair_count}"
        )

    def load_bank(self):
        with open(self.schemas_path, "r", encoding="utf-8") as f:
            self.api_schemas = json.load(f)

        with open(self.docs_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        self.documents = [
            Document(page_content=x["page_content"], metadata=x["metadata"])
            for x in raw_docs
        ]

        with open(self.bm25_path, "rb") as f:
            self.tokenized_corpus = pickle.load(f)

        self.name2schema = {x["name"]: x for x in self.api_schemas}
        self.doc_name_order = [doc.metadata["name"] for doc in self.documents]

        self.vectorstore = FAISS.load_local(
            str(self.faiss_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("✅ RAG Database Loaded!")

    # ================= [升级的核心检索接口] =================
    
    def retrieve_debug(self, query: str, top_k: int = 5):
        """
        全链路底层检索：宽进双路召回 -> CrossEncoder精排
        返回完整的信息字典，供测试脚本计算排序指标。
        """
        # 1. 宽进：多拿候选给重排模型挑选（保证至少拿 15 个）
        pool_k = max(15, top_k * 3)
        ranked = self._hybrid_rank(query, top_k=pool_k)

        ranked = ranked[:pool_k]  # 严格剔除 15 名开外的长尾噪声，防止污染重排！
        
        # 2. 精排：使用 CrossEncoder 深度打分
        if getattr(self, "reranker", None) is not None and ranked:
            pairs = [[query, f"API名称: {item['schema'].get('name', '')}\n描述: {item['schema'].get('description', '')}"] for item in ranked]
            scores = self.reranker.predict(pairs)
            
            # 将打分存入字典，并根据打分重新排序
            for i, item in enumerate(ranked):
                item["rerank_score"] = float(scores[i])
            ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # 3. 截断返回
        return ranked[:top_k]

    def retrieve_raw(self, query: str, top_k: int = 3):
        """
        基础业务检索：只返回纯净的 Schema 列表
        """
        ranked = self.retrieve_debug(query, top_k=top_k)
        return [item["schema"] for item in ranked]

    def retrieve_with_fallback(self, query: str, top_k: int = 3):
        """
        端到端路由检索：精准重排 + 常驻兜底机制
        (Web UI, RL 训练和测试的最终调用入口)
        """
        # 第一步：获取已经过重排截断的精准 Top-K
        retrieved = self.retrieve_raw(query, top_k=top_k)

        # 第二步：常驻兜底机制（如果精排前排没有，则系统强制追加到底部）
        if not any(tool["name"] == "unsupported_request" for tool in retrieved):
            fallback_tool = self.name2schema.get("unsupported_request")
            if fallback_tool is not None:
                retrieved.append(fallback_tool) 

        return retrieved


if __name__ == "__main__":
    memory = APIMemoryBank()
    memory.build_real_api_bank()
