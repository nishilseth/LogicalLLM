"""Memory-lean LoRA instruction tuning shared by the poker and chess pipelines.

Only the completion tokens contribute to the loss, so we project just the last K
positions through the LM head (a 151k-word vocab makes full-sequence logits the
dominant memory cost - this is what lets 0.5B models train on an 8GB laptop).
"""
import json, random
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from .llm import device

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def run_sft(model_name, pairs, out_dir, log_path, epochs=1, bs=2, accum=4, lr=2e-4, r=16):
    """pairs: list of (chat_messages, target_text)."""
    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()

    examples = []
    for chat, target in pairs:
        p = tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        c = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        examples.append((p + c, [-100] * len(p) + c))
    K = 1 + max(sum(t != -100 for t in x[1]) for x in examples)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = max(1, epochs * (len(examples) // (bs * accum)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps))
    rng, log, step, micro = random.Random(0), [], 0, 0
    model.train()
    for _ in range(epochs):
        rng.shuffle(examples)
        for i in range(0, len(examples) - bs + 1, bs):
            batch = examples[i:i + bs]
            L = max(len(x[0]) for x in batch)
            pad = lambda seq, v: [v] * (L - len(seq)) + seq
            ids = torch.tensor([pad(x[0], tok.pad_token_id) for x in batch], device=dev)
            lab = torch.tensor([pad(x[1], -100) for x in batch], device=dev)
            att = torch.tensor([[0] * (L - len(x[0])) + [1] * len(x[0]) for x in batch], device=dev)
            logits = model(input_ids=ids, attention_mask=att, logits_to_keep=K).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, L - K + 1:].reshape(-1), ignore_index=-100)
            (loss / accum).backward(); micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                log.append({"step": step, "loss": loss.item()})
                if step % 2 == 0:
                    print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(out_dir)
    json.dump(log, open(log_path, "w"))
    print("saved adapter ->", out_dir)
