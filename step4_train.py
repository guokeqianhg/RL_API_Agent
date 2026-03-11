import torch
import torch.distributed.fsdp

if getattr(torch.distributed.fsdp, "FSDPModule", None) is None:
    torch.distributed.fsdp.FSDPModule = type("FSDPModule", (), {})
    
from transformers import TrainerCallback, BitsAndBytesConfig
import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import other
from trl import GRPOConfig, GRPOTrainer
from step3_environment import format_reward_func, correctness_reward_func
import matplotlib.pyplot as plt

class BeautifulLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        step = state.global_step
        loss = logs.get("loss", 0.0)
        print(f"\n🚀 [Step: {step:3d}] Loss: {loss:.4f}")

def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"🚀 初始化 GRPO 训练流程，基座模型: {model_id}")
    
    with open("../data/grpo_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        row["expected_params"] = json.dumps(row.get("expected_params", {}), ensure_ascii=False)
        
    dataset = Dataset.from_list(data)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # 强烈建议 Qwen 模型使用 <|endoftext|> 作为 padding
        tokenizer.pad_token = "<|endoftext|>"
        
    # 【核心修改】：充分利用 4x4090 算力，提升探索空间，降低梯度震荡
    training_args = GRPOConfig(
        output_dir="../outputs/grpo_agent",
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        logging_steps=5,
        num_generations=8,             # <-- 从 2 改为了 8
        per_device_train_batch_size=1, # 保持不变，防止爆显存
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,   # 保持不变，节省显存
        num_train_epochs=2,            
        bf16=True,
        report_to="none"              
    )
    
    # 保持完美的 QLoRA 4-bit 配置不变
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    print("🔥 带着 4-bit 量化配置并在单卡上加载基座模型...")
    # 强制绑定到第 0 张卡，防止多卡通信损耗。如果想用其他卡，可改为 "cuda:1" 等。
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": "cuda:0"} 
    )
    model = prepare_model_for_kbit_training(model)
    
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name
    other.get_submodules = _get_submodules

    print("🔥 启动 QLoRA 训练 (已搭载裁判模型与纠错数据)...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[BeautifulLogCallback()]
    )
    
    trainer.train()
    trainer.save_model("../outputs/grpo_agent_final")
    tokenizer.save_pretrained("../outputs/grpo_agent_final")
    print("💾 训练完成，模型已保存！")

    print("📈 正在生成 Loss 曲线图...")
    log_history = trainer.state.log_history
    steps, losses = [], []
    for log in log_history:
        if "loss" in log and "step" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
            
    if steps and losses:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Training Loss')
        plt.title('GRPO Training Loss Curve (Gen=8)', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        os.makedirs("../outputs", exist_ok=True)
        plt.savefig("../outputs/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Loss 曲线已成功保存！")
        
if __name__ == "__main__":
    main()