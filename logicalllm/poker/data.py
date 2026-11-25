import random
from datasets import load_dataset

SYSTEM = ("You are an expert No-Limit Texas Hold'em player. Reply with exactly one action and nothing else: "
          "fold, check, call, bet <chips>, or raise <chips>.")


def to_chat(instruction: str):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction.strip()}]


def load_pokerbench(split="train", n=None, seed=0, balanced=True):
    """Load RZ412/PokerBench, optionally a class-balanced random subsample of size n."""
    ds = load_dataset("RZ412/PokerBench", split=split)
    if n is None or n >= len(ds):
        return ds
    rng = random.Random(seed)
    if not balanced:
        return ds.select(rng.sample(range(len(ds)), n))
    by_kind = {}
    for i, o in enumerate(ds["output"]):
        by_kind.setdefault(o.split()[0], []).append(i)
    per = n // len(by_kind)
    idx = [i for v in by_kind.values() for i in rng.sample(v, min(per, len(v)))]
    rng.shuffle(idx)
    return ds.select(idx)
