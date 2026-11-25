"""Stage 1 - instruction tuning (LoRA) on PokerBench: prompt -> single action."""
import argparse
from logicalllm.common.sft import run_sft
from .data import load_pokerbench, to_chat

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--n", type=int, default=800); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--out", default="outputs/poker-sft"); ap.add_argument("--log", default="results/poker/sft_loss.json")
a = ap.parse_args()
ds = load_pokerbench("train", n=a.n, seed=0)
run_sft(a.model, [(to_chat(i), o) for i, o in zip(ds["instruction"], ds["output"])], a.out, a.log,
        epochs=a.epochs, bs=a.bs, accum=a.accum, lr=a.lr)
