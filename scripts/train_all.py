"""Run the whole pipeline end to end (resumable: finished stages are skipped).

  python scripts/check_env.py                 # what hardware do I have?
  python scripts/setup_stockfish.py           # once, for chess
  python scripts/train_all.py --profile gpu   # NVIDIA laptop (e.g. 4-6GB VRAM)
  python scripts/train_all.py --profile mac   # 8GB Apple silicon (what the README numbers used)
  python scripts/train_all.py --profile gpu --only poker-grpo    # one stage; --force to redo

Stages: poker-base -> poker-sft -> poker-sft-eval -> poker-grpo -> poker-grpo-eval
        -> chess-sft -> chess-eval -> figures
"""
import argparse, os, subprocess, sys

PROFILES = {
    # mac = the exact settings behind the numbers in the README
    "mac": dict(sft_n=800, sft_bs=2, sft_accum=2, grpo_n=200, grpo_steps=25, grpo_group=4,
                chess_games=400, chess_bs=2, chess_accum=2, eval_n=300, eval_bs=16, positions=40, games=4),
    # gpu = bigger run for a 4-6GB NVIDIA laptop GPU; raise further if you have VRAM/time
    "gpu": dict(sft_n=8000, sft_bs=2, sft_accum=8, grpo_n=1000, grpo_steps=200, grpo_group=4,
                chess_games=3000, chess_bs=2, chess_accum=8, eval_n=500, eval_bs=16, positions=100, games=10),
}
ap = argparse.ArgumentParser()
ap.add_argument("--profile", choices=PROFILES, required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--only"); ap.add_argument("--force", action="store_true")
a = ap.parse_args()
P, M = PROFILES[a.profile], a.model
# results/ belongs to ONE profile. If it was produced by a different profile (e.g. the committed Mac
# results), archive it to results_<old>/ so stale metrics are never mistaken for this run.
import shutil
tag = "results/PROFILE"
if os.path.isdir("results") and not a.only:
    old = open(tag).read().strip() if os.path.exists(tag) else "mac"
    if old != a.profile and os.listdir("results"):
        dest = f"results_{old}"
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move("results", dest); print(f"archived previous {old} results -> {dest}/")
os.makedirs("results/poker", exist_ok=True); os.makedirs("results/chess", exist_ok=True); os.makedirs("outputs", exist_ok=True)
open(tag, "w").write(a.profile)
py = [sys.executable, "-u", "-m"]

STAGES = [
    ("poker-base", "results/poker/zeroshot.json", py + ["logicalllm.poker.evaluate", "--model", M, "-n", P["eval_n"], "--batch-size", P["eval_bs"], "--baselines", "--out", "results/poker/zeroshot.json"]),
    ("poker-sft", "outputs/poker-sft/adapter_config.json", py + ["logicalllm.poker.train_sft", "--model", M, "--n", P["sft_n"], "--bs", P["sft_bs"], "--accum", P["sft_accum"]]),
    ("poker-sft-eval", "results/poker/sft_eval.json", py + ["logicalllm.poker.evaluate", "--model", M, "--adapter", "outputs/poker-sft", "-n", P["eval_n"], "--batch-size", P["eval_bs"], "--out", "results/poker/sft_eval.json"]),
    ("poker-grpo", "outputs/poker-grpo/adapter_config.json", py + ["logicalllm.poker.train_grpo", "--model", M, "--init-adapter", "outputs/poker-sft", "--n", P["grpo_n"], "--group", P["grpo_group"], "--steps", P["grpo_steps"]]),
    ("poker-grpo-eval", "results/poker/grpo_eval.json", py + ["logicalllm.poker.evaluate", "--model", M, "--adapter", "outputs/poker-grpo", "-n", P["eval_n"], "--batch-size", P["eval_bs"], "--out", "results/poker/grpo_eval.json"]),
    ("chess-sft", "outputs/chess-sft/adapter_config.json", py + ["logicalllm.chess.train_sft", "--model", M, "--games", P["chess_games"], "--bs", P["chess_bs"], "--accum", P["chess_accum"]]),
    ("chess-eval", "results/chess/eval.json", py + ["logicalllm.chess.evaluate", "--model", M, "--adapter", "outputs/chess-sft", "--positions", P["positions"], "--games", P["games"], "--out", "results/chess/eval.json"]),
    ("figures", None, [sys.executable, "scripts/make_figures.py"]),
]
env = dict(os.environ, PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
for name, marker, cmd in STAGES:
    if a.only and a.only != name:
        continue
    if marker and os.path.exists(marker) and not a.force:
        print(f"[skip] {name} (found {marker})"); continue
    print(f"\n===== {name} =====", flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode:
        sys.exit(f"stage {name} failed (exit {r.returncode}); fix and re-run - finished stages are skipped")
print("\nAll done. Results in results/, figures in docs/figures/, adapters in outputs/")
