"""input text -> token IDs -> embeddings -> causal attention x12 -> MLP -> logits -> text"""
import argparse, os
from tokenizer import BPETokenizer
from gpt2 import GPT2

here = os.path.dirname(__file__)
ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="Poker is a game of")
ap.add_argument("--tokens", type=int, default=25)
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

tok = BPETokenizer(f"{here}/data/vocab.json", f"{here}/data/merges.txt")
model = GPT2()
ids = tok.encode(a.prompt)
print("prompt   :", repr(a.prompt))
print("token ids:", ids)
logits = model.forward(ids)
print("logits   :", logits.shape, "(positions x vocab)")
out = model.generate(ids, a.tokens, a.temperature)
print("output   :", tok.decode(out))
