"""GPT-2 (124M) forward pass in pure NumPy, loading the official weights.

Components, all written by hand:
  token + positional embeddings -> N x [LayerNorm -> multi-head *causal*
  self-attention -> residual -> LayerNorm -> MLP(GELU) -> residual]
  -> final LayerNorm -> logits via the *tied* token-embedding matrix.
"""
import json, struct
import numpy as np

def load_safetensors(path):
    """Minimal .safetensors reader: 8-byte header length, JSON header, raw bytes."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        data = np.memmap(path, dtype=np.uint8, mode="r", offset=base)
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = np.frombuffer(data[s:e], dtype=np.float32).reshape(meta["shape"])
    return out


def download_weights():
    from huggingface_hub import hf_hub_download  # download only; all math below is NumPy
    return hf_hub_download("openai-community/gpt2", "model.safetensors")


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def gelu(x):  # GPT-2's tanh approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b


def causal_self_attention(x, W_qkv, b_qkv, W_o, b_o, n_head):
    T, C = x.shape
    q, k, v = np.split(x @ W_qkv + b_qkv, 3, axis=-1)
    heads = lambda t: t.reshape(T, n_head, C // n_head).transpose(1, 0, 2)  # (H, T, d)
    q, k, v = heads(q), heads(k), heads(v)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(C // n_head)               # (H, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)                       # True = future token
    scores = np.where(mask, -1e10, scores)                                 # forbid attending ahead
    out = (softmax(scores) @ v).transpose(1, 0, 2).reshape(T, C)
    return out @ W_o + b_o


class GPT2:
    def __init__(self, weights_path=None, n_head=12):
        w = load_safetensors(weights_path or download_weights())
        self.w = {k.replace("transformer.", ""): v for k, v in w.items()}
        self.n_head = n_head
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in self.w if k.startswith("h."))

    def forward(self, ids):
        w = self.w
        ids = np.asarray(ids)
        x = w["wte.weight"][ids] + w["wpe.weight"][: len(ids)]
        for i in range(self.n_layer):
            p = f"h.{i}."
            h = layer_norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
            x = x + causal_self_attention(h, w[p + "attn.c_attn.weight"], w[p + "attn.c_attn.bias"],
                                          w[p + "attn.c_proj.weight"], w[p + "attn.c_proj.bias"], self.n_head)
            h = layer_norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
            h = gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
            x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
        return x @ w["wte.weight"].T                                       # tied output projection

    def generate(self, ids, max_new_tokens=30, temperature=0.0, top_k=40, seed=0):
        rng = np.random.default_rng(seed)
        ids = list(ids)
        for _ in range(max_new_tokens):
            logits = self.forward(ids[-1024:])[-1]
            if temperature <= 0:
                nxt = int(np.argmax(logits))                               # greedy
            else:
                logits = logits / temperature
                if top_k:
                    logits[logits < np.sort(logits)[-top_k]] = -np.inf
                nxt = int(rng.choice(len(logits), p=softmax(logits)))
            ids.append(nxt)
        return ids
