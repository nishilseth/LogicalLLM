"""Agents: pure LLM, and LLM + Stockfish ("engine-in-the-loop") wrappers.

  LLMAgent     - take the model's top move; fall back to a random legal move if illegal
  HybridAgent  - LLM proposes k candidate moves, Stockfish (shallow) picks the best
                 *legal* one. The LLM supplies intuition/pruning, the engine supplies
                 tactical verification, so illegal/blundering outputs are filtered out.
"""
import random
import chess
import torch
from .data import build_prompt, to_chat


class LLMAgent:
    name = "llm"

    def __init__(self, model, tok, k=1):
        self.model, self.tok, self.k = model, tok, k
        self.hist = []

    def candidates(self, board, history_san, k):
        text = self.tok.apply_chat_template(to_chat(build_prompt(board, history_san)), tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=10, do_sample=False, num_beams=max(k, 1),
                                      num_return_sequences=max(k, 1), pad_token_id=self.tok.pad_token_id)
        return [s.strip().split("\n")[0].split(" ")[0] for s in
                self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]

    @staticmethod
    def to_move(board, san):
        try:
            return board.parse_san(san)
        except ValueError:
            return None

    def move(self, board, history_san):
        """returns (move, info) where info['legal'] says whether the model's own first choice was legal."""
        cands = self.candidates(board, history_san, self.k)
        legal = [m for m in (self.to_move(board, c) for c in cands) if m is not None]
        if legal:
            return legal[0], {"legal": self.to_move(board, cands[0]) is not None}
        return random.choice(list(board.legal_moves)), {"legal": False}


class HybridAgent(LLMAgent):
    name = "llm+stockfish"

    def __init__(self, model, tok, engine, k=5):
        super().__init__(model, tok, k)
        self.engine = engine

    def move(self, board, history_san):
        cands = self.candidates(board, history_san, self.k)
        legal = []
        for c in cands:
            m = self.to_move(board, c)
            if m is not None and m not in legal:
                legal.append(m)
        if not legal:
            return self.engine.best_move(board), {"legal": False}
        best = max(legal, key=lambda m: self.engine.score_cp(board, m))
        return best, {"legal": self.to_move(board, cands[0]) is not None}
