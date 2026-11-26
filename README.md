# LogicalLLM - Fine-tuning LLMs for Game AI (Poker & Chess)

*Seasons of Code, Web and Coding Club, IIT Bombay (May-Jul 2025), later extended into an end-to-end project.*

Can a small language model learn to play imperfect-information (poker) and perfect-information (chess) games, and how
do you measure that honestly against solvers and engines? This repo builds the full stack, from a neural net written
in pure Python up to LoRA instruction-tuning, GRPO reinforcement learning, engine integration and an evaluation framework.

| | |
|---|---|
| **Poker** | LoRA instruction-tuning, then **GRPO** with a verifiable, solver-derived reward, on [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) |
| **Chess** | LoRA instruction-tuning on real Lichess games; **LLM + Stockfish** hybrid agent; legal-move / centipawn-loss / match evaluation |
| **From scratch** | MLP with manual backprop (pure Python, no NumPy); GPT-2 byte-level BPE tokenizer and full transformer forward pass (NumPy) |
| **Demo** | Gradio app: compare base vs fine-tuned poker decisions, play chess against the model, generate text with the NumPy GPT-2 |

## Results (small-scale runs on an 8 GB MacBook; reproducible with `--profile mac`)

All models: **Qwen2.5-0.5B-Instruct** + LoRA. Evaluation uses a fixed, class-balanced 300-hand subset of the PokerBench *test* split
(chance = 20%).

![poker results](docs/figures/poker_results.png)

| Poker policy | Action-type accuracy | Mean reward (max 2.0) |
|---|---|---|
| Chance / always-call | 20% | 0.56 |
| Zero-shot (base model) | 20% (it answers "fold" ~97% of the time) | 0.55 |
| + LoRA SFT (800 hands, 200 steps) | **44%** | 0.99 |
| + GRPO (20 steps from the SFT adapter) | 43% | 0.98 |

**Takeaways, honestly stated**
- Instruction tuning more than doubles accuracy over zero-shot (20% to 44%) with under 800 training hands.
- 20 GRPO steps did **not** improve on SFT. That's expected at this budget (tiny group size, 20 steps, lr 5e-6), and I don't claim otherwise. The GRPO pipeline itself (group sampling, z-scored advantages, clipped update, shaped reward) runs end to end; a longer run is the next experiment (`--profile gpu`, or the Colab notebook).
- Neither model ever predicts `bet`/`raise`; exact chip sizes are hard to learn, which is why the reward gives partial credit for close sizes.

![chess results](docs/figures/chess_results.png)

| Chess agent (40 held-out Lichess positions) | Legal first move | Avg. centipawn loss (lower = better) |
|---|---|---|
| Random legal move | 100% | 449 |
| LLM only (SFT on ~1,600 positions from 400 games) | 22.5% | 418 |
| LLM proposes, Stockfish verifies (k=5) | - | 250 |

- In 4 games vs. Stockfish limited to 1350 Elo, the LLM-only agent scored 0/4 and the hybrid scored 0.5/4.
- **Read the hybrid number carefully:** when none of the LLM's candidates is legal, the hybrid falls back to Stockfish, so much of its improvement is Stockfish doing the work. The honest conclusion is that a 0.5B model trained on ~1.6k positions has not learned to play chess; it has learned the move *format*. The evaluation framework is what's reusable here.
- Obvious next steps: far more data (the pipeline streams Lichess; the `gpu` profile uses ~12k positions, more is easy), put the legal-move list in the prompt, and a larger model.

## Quick start

```bash
git clone https://github.com/nishilseth/LogicalLLM.git && cd LogicalLLM
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch                                         # NVIDIA GPU: use the CUDA build, see docs/TRAINING.md
pip install -r requirements.txt
python scripts/setup_stockfish.py                         # chess engine
python -m pytest -q tests                                 # 15 tests

python demo/app.py                                        # interactive demo (needs trained adapters in outputs/)
python scripts/train_all.py --profile mac                 # or --profile gpu; full pipeline, resumable
```
Full instructions for training on an NVIDIA laptop: [docs/TRAINING.md](docs/TRAINING.md). Colab/Unsloth GRPO notebook (Gemma-3-1B): [notebooks/poker_grpo_colab.ipynb](notebooks/poker_grpo_colab.ipynb) - *written but not yet executed*.

## How it works

### Poker: SFT then GRPO ([logicalllm/poker](logicalllm/poker))
PokerBench labels come from a game-theory-optimal solver, so every prompt has a *checkable* answer - ideal for RL with verifiable rewards.
1. **SFT** teaches the output format and a first-order policy (loss on completion tokens only).
2. **GRPO** samples a *group* of G actions per hand, scores each with the reward below, and uses the **z-score of the reward within the group** as the advantage, `A_i = (r_i - mean) / std`, with no value network. PPO-style clipped updates (TRL's `GRPOTrainer`).
3. **Reward** ([reward.py](logicalllm/poker/reward.py), max 2.0): 0.2 for a well-formed single action, 1.0 for the right action type, up to 0.8 for bet/raise size closeness. Partial credit on size provides a learnable signal where exact chip counts are noise.

### Chess ([logicalllm/chess](logicalllm/chess))
- **Data:** streams the Lichess open database; keeps games with both players >= 1800 Elo and learns from the *winner's* moves. Prompt = FEN + recent moves; target = SAN move.
- **Engine integration:** [`engine.py`](logicalllm/chess/engine.py) wraps Stockfish; [`agent.py`](logicalllm/chess/agent.py) has an LLM-only agent and a hybrid that has the LLM propose k candidates (beam search) and Stockfish pick the best legal one.
- **Evaluation framework:** [`evaluate.py`](logicalllm/chess/evaluate.py) reports legal-move rate, top-1 agreement with Stockfish, average centipawn loss, and match play against Elo-limited Stockfish.

### From scratch ([foundations/](foundations))
- **MLP** ([mlp_from_scratch](foundations/mlp_from_scratch)): pure Python, manual backprop, mini-batch SGD, binary cross-entropy; 96% test accuracy on two-moons. Backprop is verified against numerical gradients in the tests.
  ![](docs/figures/mlp_decision_boundary.png)
- **Transformer** ([transformer_from_scratch](foundations/transformer_from_scratch)): GPT-2 byte-level BPE tokenizer written from scratch (matches `tiktoken` exactly on the test strings), multi-head **causal** self-attention, GELU MLP, LayerNorm, tied output embeddings, generating coherent text from the real GPT-2 weights in pure NumPy. A test proves that perturbing a future token cannot change earlier positions.
  `python foundations/transformer_from_scratch/main.py "Poker is a game of"`

### Paper notes ([docs/paper-notes.md](docs/paper-notes.md))
Written analysis of QLoRA, PokerGPT and GRPO/DeepSeekMath, including limitations and proposed extensions.

## Repo layout
```
foundations/    MLP (pure Python) and GPT-2 forward pass (NumPy)
logicalllm/     poker/ chess/ common/   - data, rewards, training, evaluation
scripts/        train_all.py  check_env.py  setup_stockfish.py  make_figures.py
demo/app.py     Gradio demo          notebooks/  Colab GRPO notebook
tests/          15 unit tests        docs/       notes, figures, training + demo guides
results/        metrics (JSON) behind every number above
```

## Limitations
- Small models and small data (hundreds to ~1.6k examples), constrained by an 8 GB laptop. The results show the pipeline working, not state-of-the-art play.
- GRPO ran for only 20 steps, with no gain over SFT. The Colab notebook has not been executed.
- Poker is scored on action type (and size credit), not on game-theoretic exploitability or win-rate in play.
- Chess evaluation uses 40 positions and 4 games, which is too small for tight confidence intervals.
