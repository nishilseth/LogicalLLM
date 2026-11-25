"""Evaluate a policy on the PokerBench test split.

Metrics: parse rate, action-type accuracy, exact-match, mean shaped reward, and
per-type accuracy. `python -m logicalllm.poker.evaluate --baselines` runs
non-LLM reference points; add --model/--adapter to score an LLM.
"""
import argparse, collections, json, random
from .actions import parse_action
from .data import load_pokerbench, to_chat
from .reward import poker_reward, MAX_REWARD


def score(preds, golds):
    n = len(golds)
    parsed = type_ok = exact = 0
    reward = 0.0
    per = collections.defaultdict(lambda: [0, 0])
    for p, g in zip(preds, golds):
        pa, ga = parse_action(p), parse_action(g)
        parsed += pa is not None
        ok = pa is not None and pa.kind == ga.kind
        type_ok += ok
        exact += pa is not None and pa == ga
        per[ga.kind][0] += ok
        per[ga.kind][1] += 1
        reward += poker_reward(p, g)
    return {"n": n, "parse_rate": parsed / n, "type_accuracy": type_ok / n, "exact_match": exact / n,
            "mean_reward": reward / n, "max_reward": MAX_REWARD,
            "per_type_accuracy": {k: v[0] / v[1] for k, v in sorted(per.items())}}


def baseline_preds(golds, name, seed=0):
    rng = random.Random(seed)
    if name == "always_call":
        return ["call"] * len(golds)
    if name == "uniform_random":
        return [rng.choice(["fold", "check", "call", "bet 10", "raise 10"]) for _ in golds]
    raise KeyError(name)


def llm_preds(prompts, model_name, adapter=None, batch_size=16, max_new_tokens=12):
    from logicalllm.common.llm import load_model, generate
    model, tok = load_model(model_name, adapter)
    chats = [to_chat(p) for p in prompts]
    return generate(model, tok, chats, batch_size=batch_size, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--adapter")
    ap.add_argument("-n", type=int, default=500); ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    ds = load_pokerbench("test", n=a.n, seed=123)  # fixed, class-balanced eval subset
    prompts, golds = ds["instruction"], ds["output"]
    results = {}
    if a.baselines:
        for b in ["always_call", "uniform_random"]:
            results[b] = score(baseline_preds(golds, b), golds)
    if a.model:
        preds = llm_preds(prompts, a.model, a.adapter, a.batch_size)
        results[a.adapter or a.model] = score(preds, golds)
        results["_samples"] = [{"pred": p, "gold": g} for p, g in list(zip(preds, golds))[:20]]
    print(json.dumps({k: v for k, v in results.items() if k != "_samples"}, indent=2))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)
