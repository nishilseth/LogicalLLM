# Trained LoRA adapters

Small LoRA adapters (~34 MB each) for `Qwen/Qwen2.5-0.5B-Instruct`, produced by `scripts/train_all.py --profile gpu`.
They are loaded by `demo/app.py` and by `--adapter models/<name>` in the evaluation scripts.

| folder | trained on | result |
|---|---|---|
| `poker-sft/` | 8,000 PokerBench hands (LoRA SFT) | 53.6% action-type accuracy |
| `poker-grpo/` | + 200 GRPO steps from `poker-sft` | 54.0% (shifts toward bet/raise) |
| `chess-sft/` | ~12k positions from 3,000 Lichess games | 24% legal first move |

Not included: base-model weights (downloaded automatically from Hugging Face).
