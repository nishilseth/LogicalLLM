"""Lichess PGNs -> chess instruction-tuning examples.

Each example is one position from a real game: the model sees the FEN plus the
last few moves and must output the move the (stronger / winning) player chose.
"""
import io, random
import chess, chess.pgn

SYSTEM = ("You are a strong chess engine. Given a position, reply with the single best legal move "
          "in standard algebraic notation (SAN) and nothing else.")


def build_prompt(board: chess.Board, history_san: list, max_plies=12) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    recent = history_san[-max_plies:]
    start = len(history_san) - len(recent)
    parts = []
    for i, san in enumerate(recent):
        ply = start + i
        parts.append(f"{ply // 2 + 1}. {san}" if ply % 2 == 0 else (san if i else f"{ply // 2 + 1}... {san}"))
    return f"FEN: {board.fen()}\nSide to move: {side}\nRecent moves: {' '.join(parts) or '(start)'}\nBest move:"


def to_chat(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def positions_from_game(game, per_game=4, min_ply=8, winner_only=True, rng=random):
    """Sample up to `per_game` positions; only moves by the winner if winner_only."""
    result = game.headers.get("Result")
    winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(result)
    if winner_only and winner is None:
        return []
    board, hist, cand = game.board(), [], []
    for mv in game.mainline_moves():
        san = board.san(mv)
        if len(hist) >= min_ply and (not winner_only or board.turn == winner):
            cand.append((build_prompt(board, hist), san))
        board.push(mv)
        hist.append(san)
    return rng.sample(cand, min(per_game, len(cand)))


def stream_lichess(n_games=1000, min_elo=1800, per_game=4, seed=0):
    """Stream games from the Lichess/standard-chess-games HF dataset (no huge download)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    games = 0
    for row in ds:
        try:
            if min(int(row["WhiteElo"]), int(row["BlackElo"])) < min_elo or row["Termination"] != "Normal":
                continue
        except (TypeError, ValueError):
            continue
        game = chess.pgn.read_game(io.StringIO(
            f'[Result "{row["Result"]}"]\n\n{row["movetext"]}'))
        if game is None:
            continue
        ex = positions_from_game(game, per_game, rng=rng)
        if ex:
            games += 1
            for p, m in ex:
                yield {"prompt": p, "move": m}
        if games >= n_games:
            return


def load_eval_positions(n, seed=999, cache="results/chess/eval_positions.json", retries=6):
    """Held-out positions (a different shuffle seed from training). Cached on disk and retried,
    because streaming from the Hub can time out on flaky connections."""
    import json, os, time
    if os.path.exists(cache):
        cached = json.load(open(cache))
        if len(cached) >= n:
            return cached[:n]
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    for attempt in range(1, retries + 1):
        try:
            ex = list(stream_lichess(n_games=n, per_game=1, seed=seed))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(ex, open(cache, "w"))
            return ex
        except Exception as e:  # network errors surface as many different exception types
            print(f"[eval positions] attempt {attempt}/{retries} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError("could not download Lichess positions; check your connection and re-run")
