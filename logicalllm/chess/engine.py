"""Thin Stockfish wrapper (via python-chess) used for labelling, re-ranking and scoring."""
import os, shutil
import chess, chess.engine

DEFAULT_PATHS = [os.environ.get("STOCKFISH_PATH", ""), shutil.which("stockfish") or "",
                 os.path.join(os.path.dirname(__file__), "..", "..", "tools", "stockfish", "stockfish-macos-universal")]
MATE_CP = 10000


def find_stockfish():
    for p in DEFAULT_PATHS:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    tools = os.path.join(os.path.dirname(__file__), "..", "..", "tools")   # scripts/setup_stockfish.py puts it here
    for root, _, files in os.walk(tools):
        for f in files:
            if f.lower().startswith("stockfish") and not f.endswith((".txt", ".md", ".zip", ".tar", ".gz")) and os.access(os.path.join(root, f), os.X_OK):
                return os.path.abspath(os.path.join(root, f))
    raise FileNotFoundError("Stockfish not found. Run python scripts/setup_stockfish.py or set STOCKFISH_PATH.")


class Stockfish:
    def __init__(self, depth=8, path=None, skill_elo=None):
        self.eng = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())
        self.depth = depth
        if skill_elo:  # weakened opponent for matches
            self.eng.configure({"UCI_LimitStrength": True, "UCI_Elo": int(skill_elo)})

    def score_cp(self, board, move=None, depth=None):
        """Centipawn eval from the *mover's* perspective, optionally after playing `move`."""
        mover = board.turn
        b = board.copy()
        if move is not None:
            b.push(move)
        info = self.eng.analyse(b, chess.engine.Limit(depth=depth or self.depth))
        return info["score"].pov(mover).score(mate_score=MATE_CP)

    def best_move(self, board, depth=None, time=None):
        lim = chess.engine.Limit(time=time) if time else chess.engine.Limit(depth=depth or self.depth)
        return self.eng.play(board, lim).move

    def close(self):
        self.eng.quit()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
