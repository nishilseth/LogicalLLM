"""Verifiable rewards for GRPO on PokerBench.

PokerBench labels come from a game-theory-optimal solver, so every prompt has a
checkable target. We shape the reward in three parts (max 2.0):

  format  0.2  completion is exactly one well-formed action
  type    1.0  action type (fold/check/call/bet/raise) equals the solver's
  size    0.8  scaled by how close the bet/raise size is (1 - relative error);
               full credit for non-sized actions once the type is right

Partial credit on size matters: exact chip amounts are nearly unlearnable noise,
whereas "bet roughly the right fraction of pot" is a real, learnable signal.
"""
from .actions import parse_action, is_strict_format

W_FORMAT, W_TYPE, W_SIZE = 0.2, 1.0, 0.8
MAX_REWARD = W_FORMAT + W_TYPE + W_SIZE


def size_closeness(pred, gold):
    if pred is None or gold is None:
        return 0.0
    return max(0.0, 1.0 - abs(pred - gold) / max(pred, gold, 1e-9))


def poker_reward(completion: str, gold: str) -> float:
    p, g = parse_action(completion), parse_action(gold)
    if g is None:
        raise ValueError(f"unparseable gold label: {gold!r}")
    r = W_FORMAT if is_strict_format(completion) else 0.0
    if p is None or p.kind != g.kind:
        return r
    r += W_TYPE
    r += W_SIZE * (size_closeness(p.amount, g.amount) if g.amount is not None else 1.0)
    return r


def grpo_reward_fn(prompts, completions, gold, **_):
    """Signature expected by TRL's GRPOTrainer. `gold` is a dataset column."""
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [poker_reward(t, g) for t, g in zip(texts, gold)]
