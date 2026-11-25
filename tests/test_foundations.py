import math, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, f"{ROOT}/foundations/mlp_from_scratch")
sys.path.insert(0, f"{ROOT}/foundations/transformer_from_scratch")
from mlp import MLP, bce            # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
import gpt2                         # noqa: E402

TOK = BPETokenizer(f"{ROOT}/foundations/transformer_from_scratch/data/vocab.json",
                   f"{ROOT}/foundations/transformer_from_scratch/data/merges.txt")

TEXTS = ["Hello, world!", "  leading   spaces", "Poker: raise 2.5bb on the BTN.",
         "naïve café — 日本語 🙂", "def f(x):\n    return x**2\n", "don't we've I'll"]


@pytest.mark.parametrize("text", TEXTS)
def test_tokenizer_matches_tiktoken_and_roundtrips(text):
    tiktoken = pytest.importorskip("tiktoken")
    ref = tiktoken.get_encoding("gpt2").encode(text)
    assert TOK.encode(text) == ref
    assert TOK.decode(TOK.encode(text)) == text


def test_mlp_backprop_matches_numerical_gradient():
    m = MLP([2, 4, 3, 1], seed=1)
    x, y = [0.3, -0.7], 1
    p, acts = m.forward(x, cache=True)
    gW = [[[0.0] * len(r) for r in W] for W in m.W]
    gb = [[0.0] * len(b) for b in m.b]
    m._backward(y, acts, gW, gb)
    eps = 1e-6
    for L in range(len(m.W)):
        for i in range(len(m.W[L])):
            for j in range(len(m.W[L][i])):
                m.W[L][i][j] += eps; up = bce(m.forward(x), y)
                m.W[L][i][j] -= 2 * eps; dn = bce(m.forward(x), y)
                m.W[L][i][j] += eps
                assert math.isclose(gW[L][i][j], (up - dn) / (2 * eps), rel_tol=1e-4, abs_tol=1e-7)


def test_mlp_learns_xor():
    m = MLP([2, 4, 1], seed=3)
    X, Y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    for _ in range(3000):
        m.train_batch(X, Y, 0.5)
    assert [m.predict(x) for x in X] == Y


def test_attention_is_causal():
    rng = np.random.default_rng(0)
    C, T, H = 16, 6, 4
    args = (rng.normal(size=(C, 3 * C)), rng.normal(size=3 * C), rng.normal(size=(C, C)), rng.normal(size=C), H)
    x = rng.normal(size=(T, C))
    base = gpt2.causal_self_attention(x, *args)
    x2 = x.copy(); x2[-1] += 100.0          # perturb only the LAST token
    out2 = gpt2.causal_self_attention(x2, *args)
    assert np.allclose(base[:-1], out2[:-1])  # earlier positions must be unaffected
    assert not np.allclose(base[-1], out2[-1])
