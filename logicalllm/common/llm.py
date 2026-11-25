"""Shared HF model loading + batched greedy generation (CUDA / Apple MPS / CPU)."""
import torch


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device()
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model.eval(), tok


@torch.no_grad()
def generate(model, tok, chats, batch_size=16, max_new_tokens=12):
    """chats: list of chat-message lists. Greedy decoding; returns list[str]."""
    out = []
    for i in range(0, len(chats), batch_size):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]
