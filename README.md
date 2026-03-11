# RL_API_Agent
# Self-Evolving MCP Agent Framework

本项目提供了一个工业级、端到端的自进化 AI 智能体（Agent）框架。系统集成了混合 RAG 工具检索机制、自动化数据合成，以及基于本地大语言模型作为裁判的强化学习（GRPO），以确保模型具备极高的指令遵循能力和强大的域外（OOD）请求拒绝能力。

## 核心特性

* **混合工具检索 (RAG)**：结合 FAISS（稠密检索）与 BM25（稀疏检索），并采用倒数秩融合（RRF）算法，根据用户查询精准召回 API 工具。
* **自动化数据合成**：调用外部大语言模型 API，根据 API Schema 自动生成贴合真实业务场景的正样本数据与用于对抗训练的域外（OOD）负样本数据。
* **本地 LLM 裁判环境**：引入轻量级本地模型（如 Qwen2.5-0.5B-Instruct）作为沙盒环境裁判，对参数提取的语义一致性进行打分并计算强化学习的 Reward。
* **GRPO 强化学习**：采用组相对策略优化（GRPO）算法与 QLoRA 4-bit 量化技术，在有限显存下对基座模型（如 Qwen2.5-7B-Instruct）进行微调。
* **工业级评估体系**：包含路由准确率、参数提取成功率、幻觉率以及域外拒答率等多维度的细粒度评估脚本。

## 项目结构

* `step1_build_rag.py`: 解析原始 API CSV 数据，构建 FAISS 与 BM25 混合向量检索库。
* `step2_make_data.py`: 基于 API 列表，自动合成训练与测试数据（包含正常意图与诱导犯错的 OOD 意图）。
* `step3_environment.py`: 定义强化学习的模拟沙盒环境与奖励函数（Reward Function），内置语义裁判模型。
* `step4_train.py`: 执行基于 QLoRA 的 GRPO 强化学习训练流程。
* `step5_inference.py`: 命令行推理脚本，用于加载 LoRA 权重并进行单轮或多轮任务测试。（暂不开源）
* `step6_web_ui.py`: 基于 Gradio 的可视化 Web 交互界面。（暂不开源）
* `step7_evaluate.py`: 端到端评估脚本，输出系统核心指标与业务逻辑混淆矩阵。（暂不开源）
* `test_RAG.py`: 测试检索系统的准确率（Hit@1, Hit@3）的验证脚本。

## 依赖安装

请确保系统已安装 Python 3.8 及以上版本。执行以下命令安装所需依赖：

```bash
pip install torch transformers peft trl datasets
pip install langchain langchain-community langchain-huggingface
pip install faiss-cpu rank_bm25 jieba pandas
pip install openai gradio matplotlib tqdm

使用指南
1. 构建 RAG 知识库
将原始 API 数据放置在 ../data/raw_api_bank/all_apis.csv。然后运行以下命令构建检索索引：
python step1_build_rag.py
可以通过运行 python test_RAG.py 来验证检索系统的准确性。

2. 合成训练数据
在调用外部 API 生成数据前，请先配置环境变量。以阿里云 DashScope 为例：
export DASHSCOPE_API_KEY="在这里填入你真实的_API_KEY"
python step2_make_data.py
执行完毕后，会在 ../data 目录下生成 grpo_train.json 和 grpo_test.json。

3. 训练智能体
启动 GRPO 强化学习流程。该脚本默认配置为 4-bit QLoRA 以优化显存占用：
python step4_train.py
训练完成后的 LoRA 权重将保存在 ../outputs/grpo_agent_final 目录。

4. 推理与交互
你可以通过命令行或 Web 界面与训练好的智能体进行交互：
命令行模式：
python step5_inference.py

Web UI 模式 (Gradio)：
python step6_web_ui.py

5. 评估模型
运行综合评估脚本，获取路由准确率、OOD 拒答率和参数提取等详细指标报告：
python step7_evaluate.py
