# Training on your own machine (NVIDIA laptop GPU)

Everything is driven by `scripts/train_all.py`. Stages are resumable: a finished stage is skipped, so if the laptop
sleeps or a run crashes, just run the same command again.

## 1. Set up (once)

```bash
git clone https://github.com/nishilseth/LogicalLLM.git && cd LogicalLLM
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# PyTorch with CUDA FIRST (pick the command for your driver at https://pytorch.org/get-started/locally/):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python scripts/setup_stockfish.py      # chess engine, downloads for your OS
python scripts/check_env.py            # should print your GPU + "recommended profile: gpu"
python -m pytest -q tests              # 15 tests should pass
```

If `check_env.py` says "NO GPU FOUND", torch was installed without CUDA - reinstall using the command above.

## 2. Train everything

```bash
python scripts/train_all.py --profile gpu
```

| stage | what it does | gpu profile |
|---|---|---|
| poker-base | zero-shot + chance baselines | 500 test hands |
| poker-sft | LoRA instruction tuning on PokerBench | 8,000 hands |
| poker-grpo | GRPO from the SFT adapter, solver-based reward | 200 steps, group of 4 |
| chess-sft | LoRA tuning on Lichess winners' moves | 3,000 games (~12k positions) |
| chess-eval | legal-move rate, centipawn loss, games vs Stockfish | 100 positions, 10 games |
| figures | writes `docs/figures/*.png` | |

Run one stage: `--only poker-grpo`; redo one: `--only poker-grpo --force`.
Adapters land in `outputs/`, metrics in `results/`, plots in `docs/figures/`.

### If something goes wrong
- **CUDA out of memory** (4 GB cards): edit the `gpu` profile in `scripts/train_all.py` and set `sft_bs`/`chess_bs` to 1 and raise the matching `accum`. For GRPO, `grpo_group` must stay at 2 or more.
- **Too slow**: lower `sft_n`, `grpo_steps`, `chess_games`. The `mac` profile is the tiny version.
- **Windows "Stockfish not found"**: `set STOCKFISH_PATH=C:\path\to\stockfish.exe`.
- Keep the laptop plugged in and disable sleep while it trains.

## 3. Look at results / demo

```bash
python demo/app.py      # open the printed http://127.0.0.1:7860
```

## 4. Push the numbers back

`results/*.json` and `docs/figures/*.png` are what the README quotes. After the GPU run, commit them and update the
README results table with your numbers.
