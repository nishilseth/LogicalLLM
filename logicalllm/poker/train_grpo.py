"""Stage 2 - GRPO (Group Relative Policy Optimisation) on PokerBench.

For every prompt the policy samples a *group* of G candidate actions. Each is
scored with the verifiable solver-based reward (reward.py); advantages are the
z-score of each reward inside its group -  A_i = (r_i - mean(r)) / std(r)  -
so no value network / critic is needed. Updates are PPO-style (clipped ratio)
with an optional KL penalty to the reference policy. TRL's GRPOTrainer
implements exactly this; we supply the dataset, reward and LoRA config.
"""
import argparse, json
import torch
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from logicalllm.common.llm import device
from .data import load_pokerbench, to_chat
from .reward import grpo_reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--init-adapter", help="start from the SFT LoRA adapter (recommended)")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    ap.add_argument("--out", default="outputs/poker-grpo")
    a = ap.parse_args()

    dev = device()
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    ds = load_pokerbench("train", n=a.n, seed=1)
    ds = ds.map(lambda r: {"prompt": to_chat(r["instruction"]), "gold": r["output"]},
                remove_columns=ds.column_names)
    cfg = GRPOConfig(output_dir=a.out, num_generations=a.group, per_device_train_batch_size=a.group,
                     gradient_accumulation_steps=2, max_completion_length=12, max_steps=a.steps,
                     learning_rate=a.lr, beta=a.beta, temperature=1.0, logging_steps=1,
                     save_strategy="no", report_to="none", bf16=False, fp16=False, use_cpu=(dev == "cpu"),
                     remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, reward_funcs=grpo_reward_fn, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(a.out)
    import os; os.makedirs("results/poker", exist_ok=True)
    json.dump(trainer.state.log_history, open("results/poker/grpo_log.json", "w"))
    print("saved adapter ->", a.out)


if __name__ == "__main__":
    main()
