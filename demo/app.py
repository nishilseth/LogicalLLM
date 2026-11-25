"""LogicalLLM demo.  Run:  python demo/app.py   (then open the printed local URL)

Tab 1  Poker : pick a PokerBench test hand; compare base model vs fine-tuned (LoRA) vs solver answer.
Tab 2  Chess : play against the LLM (+ optional Stockfish verification) on a board.
Tab 3  From scratch : pure-NumPy GPT-2 text generation.
"""
import os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/foundations/transformer_from_scratch")
import chess, chess.svg
import gradio as gr

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
POKER_ADAPTER = os.environ.get("POKER_ADAPTER", f"{ROOT}/outputs/poker-sft")
CHESS_ADAPTER = os.environ.get("CHESS_ADAPTER", f"{ROOT}/outputs/chess-sft")
_cache = {}


def get_model(adapter):
    key = adapter if adapter and os.path.exists(adapter) else None
    if key not in _cache:
        from logicalllm.common.llm import load_model
        _cache[key] = load_model(BASE, key)
    return _cache[key], key is not None


# ---------------- poker ----------------
_poker = {}


def poker_samples():
    if not _poker:
        from logicalllm.poker.data import load_pokerbench
        _poker["ds"] = load_pokerbench("test", n=300, seed=7)
    return _poker["ds"]


def poker_new():
    ds = poker_samples(); i = random.randrange(len(ds))
    return ds[i]["instruction"].strip(), ds[i]["output"], "", "", ""


def poker_run(scenario, gold):
    from logicalllm.common.llm import generate
    from logicalllm.poker.data import to_chat
    from logicalllm.poker.reward import poker_reward
    res = []
    for adapter in (None, POKER_ADAPTER):
        (model, tok), is_ft = get_model(adapter)
        if adapter and not is_ft:
            res.append("(no fine-tuned adapter found - run scripts/run_all.sh)"); continue
        a = generate(model, tok, [to_chat(scenario)], max_new_tokens=12)[0]
        res.append(f"{a}   (reward {poker_reward(a, gold):.2f}/2.00)" if gold else a)
    return res[0], res[1], gold


# ---------------- chess ----------------
def board_svg(b, last=None):
    return chess.svg.board(b, lastmove=last, size=420)


def chess_new():
    b = chess.Board()
    return b.fen(), board_svg(b), "Your move (you are White). Enter SAN like e4 or Nf3."


def chess_play(fen, move_san, use_engine):
    from logicalllm.chess.agent import LLMAgent, HybridAgent
    b = chess.Board(fen)
    try:
        b.push_san(move_san.strip())
    except ValueError:
        return fen, board_svg(b), f"Illegal move: {move_san}"
    if b.is_game_over():
        return b.fen(), board_svg(b), f"Game over: {b.result()}"
    (model, tok), is_ft = get_model(CHESS_ADAPTER)
    if use_engine:
        from logicalllm.chess.engine import Stockfish
        eng = Stockfish(depth=6)
        try:
            mv, _ = HybridAgent(model, tok, eng, k=5).move(b, [])
        finally:
            eng.close()
    else:
        mv, info = LLMAgent(model, tok).move(b, [])
    san = b.san(mv); b.push(mv)
    note = "" if is_ft else " [base model - chess adapter not found]"
    status = f"Model played {san}.{note}" if not b.is_game_over() else f"Game over: {b.result()}"
    return b.fen(), board_svg(b, mv), status


# ---------------- scratch gpt2 ----------------
def scratch(prompt, n):
    from tokenizer import BPETokenizer
    from gpt2 import GPT2
    if "g" not in _cache:
        d = f"{ROOT}/foundations/transformer_from_scratch/data"
        _cache["g"] = (BPETokenizer(f"{d}/vocab.json", f"{d}/merges.txt"), GPT2())
    tok, g = _cache["g"]
    ids = tok.encode(prompt)
    return f"token ids: {ids}\n\n" + tok.decode(g.generate(ids, int(n)))


with gr.Blocks(title="LogicalLLM") as demo:
    gr.Markdown("# LogicalLLM - fine-tuning LLMs for game AI (poker & chess)")
    with gr.Tab("Poker (GRPO / SFT on PokerBench)"):
        scen = gr.Textbox(label="Scenario (PokerBench test hand)", lines=14)
        gold = gr.Textbox(label="Solver-optimal action", interactive=False)
        with gr.Row():
            b1 = gr.Button("New random hand"); b2 = gr.Button("Ask the models", variant="primary")
        with gr.Row():
            o1 = gr.Textbox(label="Base model (zero-shot)"); o2 = gr.Textbox(label="Fine-tuned model")
        b1.click(poker_new, outputs=[scen, gold, o1, o2, gr.Textbox(visible=False)])
        b2.click(poker_run, [scen, gold], [o1, o2, gold])
    with gr.Tab("Chess"):
        fen = gr.Textbox(label="FEN", interactive=False); svg = gr.HTML(); msg = gr.Markdown()
        mv = gr.Textbox(label="Your move (SAN)"); eng = gr.Checkbox(label="Let Stockfish verify the LLM's candidate moves", value=True)
        with gr.Row():
            n = gr.Button("New game"); p = gr.Button("Play", variant="primary")
        n.click(chess_new, outputs=[fen, svg, msg]); p.click(chess_play, [fen, mv, eng], [fen, svg, msg])
        demo.load(chess_new, outputs=[fen, svg, msg])
    with gr.Tab("From scratch (NumPy GPT-2)"):
        pr = gr.Textbox(label="Prompt", value="Poker is a game of"); nt = gr.Slider(5, 40, 20, step=1, label="new tokens")
        out = gr.Textbox(label="Output", lines=8); gr.Button("Generate").click(scratch, [pr, nt], out)

if __name__ == "__main__":
    demo.launch()
