"""Instruction-tune (LoRA) an LLM on moves from real Lichess games (winner's moves only)."""
import argparse, json, os
from logicalllm.common.sft import run_sft
from .data import stream_lichess, to_chat

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--games", type=int, default=400, help="games to sample (use 50000+ on a GPU)")
ap.add_argument("--per-game", type=int, default=4); ap.add_argument("--min-elo", type=int, default=1800)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--out", default="outputs/chess-sft")
a = ap.parse_args()
os.makedirs("results/chess", exist_ok=True)
ex = list(stream_lichess(a.games, a.min_elo, a.per_game, seed=0))
print(f"{len(ex)} training positions from {a.games} games", flush=True)
json.dump(ex[:200], open("results/chess/train_sample.json", "w"), indent=1)
run_sft(a.model, [(to_chat(e["prompt"]), e["move"]) for e in ex], a.out, "results/chess/sft_loss.json",
        epochs=a.epochs, bs=a.bs, accum=a.accum)
