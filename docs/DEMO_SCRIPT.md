# Demo video script (~3 min)

Record with QuickTime (Mac) or Win+G (Windows). Run `python demo/app.py` first and keep the terminal and browser side by side.

| time | show | say |
|---|---|---|
| 0:00 | README top + architecture diagram | "LogicalLLM: I fine-tuned small LLMs to play poker and chess, and benchmarked them against solvers and engines." |
| 0:20 | Poker tab: click **New random hand**, then **Ask the models** | "Left is the base model: it just says fold. Right is the model after LoRA tuning and GRPO. Under it is the solver's optimal action." Repeat 2-3 hands. |
| 1:00 | `docs/figures/poker_results.png` | "Zero-shot is chance, 20%. After tuning, accuracy goes up. Bet and raise are hardest because exact sizing is nearly unlearnable, so the reward gives partial credit for close sizes." |
| 1:30 | Chess tab: play 1.e4, tick the Stockfish checkbox | "The LLM proposes candidate moves; Stockfish checks them, so illegal moves and blunders get filtered." |
| 2:15 | From-scratch tab: generate from "Poker is a game of" | "This GPT-2 is pure NumPy: my own BPE tokenizer, causal attention, MLP and logits. The tokenizer matches the reference exactly." |
| 2:45 | `pytest` output | "All components are unit-tested." |

Be honest in the narration about scale (small models, small data) - interviewers respect it.
