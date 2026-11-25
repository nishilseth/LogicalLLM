"""Evaluation framework for chess agents.

1) Position benchmark  - on held-out positions (Lichess games the model never saw):
     legal-move rate, top-1 agreement with Stockfish, average centipawn loss (ACPL)
2) Match play          - agent vs. Stockfish limited to a given Elo; score + Elo-diff estimate
"""
import argparse, json, math, random
import chess
from .engine import Stockfish
from .agent import LLMAgent, HybridAgent
from .data import stream_lichess


def position_benchmark(agent, engine, examples):
    n = legal = top1 = 0
    losses = []
    for ex in examples:
        board = chess.Board(ex["prompt"].split("\n")[0][5:])
        hist = []  # history not needed for scoring; agent prompt uses FEN
        move, info = agent.move(board, hist)
        n += 1
        legal += info["legal"]
        best = engine.best_move(board)
        top1 += move == best
        losses.append(min(1000, max(0, engine.score_cp(board, best) - engine.score_cp(board, move))))
    return {"positions": n, "legal_rate": legal / n, "top1_vs_stockfish": top1 / n,
            "acpl": sum(losses) / n}


def play_game(agent, engine, agent_white, max_plies=120):
    board, hist = chess.Board(), []
    illegal = 0
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if (board.turn == chess.WHITE) == agent_white:
            mv, info = agent.move(board, hist)
            illegal += not info["legal"]
        else:
            mv = engine.best_move(board, time=0.02)
        hist.append(board.san(mv))
        board.push(mv)
    res = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    pts = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    return (pts if agent_white else 1 - pts), illegal, len(board.move_stack)


def match(agent, opp_elo, games=6):
    pts, ill, plies = 0.0, 0, 0
    with Stockfish(skill_elo=opp_elo) as opp:
        for g in range(games):
            p, i, n = play_game(agent, opp, agent_white=(g % 2 == 0))
            pts += p; ill += i; plies += n
    s = min(max(pts / games, 0.01), 0.99)
    return {"opponent_elo": opp_elo, "games": games, "score": pts / games,
            "elo_diff_estimate": round(-400 * math.log10(1 / s - 1)), "illegal_moves": ill, "plies": plies}


if __name__ == "__main__":
    from logicalllm.common.llm import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--adapter")
    ap.add_argument("--positions", type=int, default=50); ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--opp-elo", type=int, default=1350); ap.add_argument("--hybrid-k", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    model, tok = load_model(a.model, a.adapter)
    # held-out: different shuffle seed from training data
    test = []
    for ex in stream_lichess(n_games=a.positions, per_game=1, seed=999):
        test.append(ex)
    res = {}
    with Stockfish(depth=8) as eng:
        for agent in (LLMAgent(model, tok, k=1), HybridAgent(model, tok, eng, k=a.hybrid_k)):
            r = {"position_benchmark": position_benchmark(agent, eng, test)}
            if a.games:
                r["match"] = match(agent, a.opp_elo, a.games)
            res[agent.name] = r
            print(agent.name, json.dumps(r, indent=1), flush=True)
        # reference: stockfish itself would be 100% top-1; random-legal for context
        rnd, ls = 0, []
        for ex in test:
            b = chess.Board(ex["prompt"].split("\n")[0][5:]); m = random.choice(list(b.legal_moves))
            ls.append(min(1000, max(0, eng.score_cp(b, eng.best_move(b)) - eng.score_cp(b, m))))
        res["random_legal"] = {"acpl": sum(ls) / len(ls)}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
