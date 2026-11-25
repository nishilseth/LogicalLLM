import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import chess, chess.pgn, io
from logicalllm.chess.data import build_prompt, positions_from_game


def test_prompt_contains_fen_and_moves():
    b = chess.Board(); b.push_san("e4"); b.push_san("e5")
    p = build_prompt(b, ["e4", "e5"])
    assert b.fen() in p and "1. e4 e5" in p and "White" in p


def test_positions_are_winner_moves_and_legal():
    g = chess.pgn.read_game(io.StringIO("[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0"))
    ex = positions_from_game(g, per_game=3, min_ply=8, rng=random.Random(0))
    assert ex
    for prompt, move in ex:
        b = chess.Board(prompt.split("\n")[0][5:])
        assert b.turn == chess.WHITE            # winner = White
        b.parse_san(move)                  # must be legal
