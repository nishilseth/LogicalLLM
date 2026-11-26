"""Build README figures from results/*.json (whatever exists)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "docs/figures"
os.makedirs(F, exist_ok=True)
load = lambda p: json.load(open(f"{R}/{p}")) if os.path.exists(f"{R}/{p}") else None
smooth = lambda y, k=10: [sum(y[max(0, i - k + 1):i + 1]) / len(y[max(0, i - k + 1):i + 1]) for i in range(len(y))]

# poker: accuracy by stage
stages = [("Chance", {"type_accuracy": 0.2, "mean_reward": None}), ]
for name, f in [("Zero-shot", "poker/zeroshot.json"), ("+ SFT (LoRA)", "poker/sft_eval.json"), ("+ GRPO", "poker/grpo_eval.json")]:
    d = load(f)
    if d:
        stages.append((name, [v for k, v in d.items() if not k.startswith("_")][0]))
if len(stages) > 1:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar([s[0] for s in stages], [s[1]["type_accuracy"] * 100 for s in stages], color=["#bbb"] + ["#4C78A8"] * (len(stages) - 1))
    ax[0].set_ylabel("Action-type accuracy (%)"); ax[0].set_title("PokerBench (balanced test subset)")
    for i, s in enumerate(stages): ax[0].text(i, s[1]["type_accuracy"] * 100 + 1, f'{s[1]["type_accuracy"]*100:.0f}%', ha="center")
    ks = ["fold", "check", "call", "bet", "raise"]; w = 0.8 / (len(stages) - 1)
    for j, s in enumerate(stages[1:]):
        ax[1].bar([i + j * w for i in range(5)], [s[1]["per_type_accuracy"].get(k, 0) * 100 for k in ks], w, label=s[0])
    ax[1].set_xticks([i + 0.4 - w / 2 for i in range(5)]); ax[1].set_xticklabels(ks); ax[1].legend(); ax[1].set_title("Accuracy per action type")
    plt.tight_layout(); plt.savefig(f"{F}/poker_results.png", dpi=140); plt.close()

sft = load("poker/sft_loss.json")
grpo = load("poker/grpo_log.json")
if sft or grpo:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    if sft:
        y = [x["loss"] for x in sft]; ax[0].plot(y, alpha=.3); ax[0].plot(smooth(y), color="C0"); ax[0].set_title("SFT loss (completion tokens)"); ax[0].set_xlabel("optimizer step")
    if grpo:
        h = [x for x in grpo if "reward" in x]
        y = [x["reward"] for x in h]; ax[1].plot(y, alpha=.3); ax[1].plot(smooth(y), color="C1")
        ax[1].set_title("GRPO mean reward (max 2.0)"); ax[1].set_xlabel("step")
    plt.tight_layout(); plt.savefig(f"{F}/poker_training.png", dpi=140); plt.close()
print("figures written to", F)

ch = load("chess/eval.json")
if ch:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    names = ["LLM only", "LLM + Stockfish", "Random legal"]
    acpl = [ch["llm"]["position_benchmark"]["acpl"], ch["llm+stockfish"]["position_benchmark"]["acpl"], ch["random_legal"]["acpl"]]
    ax[0].bar(names, acpl, color=["#4C78A8", "#F58518", "#bbb"]); ax[0].set_title("Avg. centipawn loss vs Stockfish (lower = better)")
    for i, v in enumerate(acpl): ax[0].text(i, v + 5, f"{v:.0f}", ha="center")
    lr = [ch["llm"]["position_benchmark"]["legal_rate"] * 100]
    ax[1].bar(["LLM first choice\nis legal (%)"], lr, color="#4C78A8"); ax[1].set_ylim(0, 100); ax[1].set_title("Legal-move rate")
    ax[1].text(0, lr[0] + 2, f"{lr[0]:.0f}%", ha="center")
    plt.tight_layout(); plt.savefig(f"{F}/chess_results.png", dpi=140); plt.close()
