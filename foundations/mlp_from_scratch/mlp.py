"""A multi-layer perceptron in pure Python (no NumPy): forward pass, manual
backpropagation, and mini-batch SGD with binary cross-entropy loss."""
import math
import random


def sigmoid(z):
    # Numerically stable: never calls exp() on a large positive number.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def bce(p, y, eps=1e-12):
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


class MLP:
    """Input -> [hidden layers, tanh] -> 1 sigmoid output.

    Using BCE with a sigmoid output, the gradient at the output pre-activation
    simplifies to (p - y), which is what `_backward` exploits.
    """

    def __init__(self, layer_sizes, seed=0):
        self.rng = random.Random(seed)
        self.sizes = list(layer_sizes)
        assert self.sizes[-1] == 1, "binary classifier: last layer must have 1 unit"
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            limit = math.sqrt(6.0 / (n_in + n_out))  # Xavier/Glorot uniform
            self.W.append([[self.rng.uniform(-limit, limit) for _ in range(n_in)] for _ in range(n_out)])
            self.b.append([0.0] * n_out)

    # ---- forward ---------------------------------------------------------
    def forward(self, x, cache=False):
        a = list(x)
        acts = [a]
        for layer, (W, b) in enumerate(zip(self.W, self.b)):
            z = [sum(w * ai for w, ai in zip(row, a)) + bi for row, bi in zip(W, b)]
            last = layer == len(self.W) - 1
            a = [sigmoid(v) for v in z] if last else [math.tanh(v) for v in z]
            acts.append(a)
        return (a[0], acts) if cache else a[0]

    def predict(self, x):
        return 1 if self.forward(x) >= 0.5 else 0

    # ---- backward --------------------------------------------------------
    def _backward(self, y, acts, gW, gb):
        """Accumulate dLoss/dW and dLoss/db for one sample into gW, gb."""
        delta = [acts[-1][0] - y]  # dL/dz at the output (BCE + sigmoid)
        for layer in range(len(self.W) - 1, -1, -1):
            a_prev = acts[layer]
            for i, d in enumerate(delta):
                gb[layer][i] += d
                row = gW[layer][i]
                for j, aj in enumerate(a_prev):
                    row[j] += d * aj
            if layer > 0:
                # propagate through W, then through tanh'(z) = 1 - a^2
                new_delta = []
                for j in range(len(a_prev)):
                    s = sum(self.W[layer][i][j] * delta[i] for i in range(len(delta)))
                    new_delta.append(s * (1.0 - a_prev[j] ** 2))
                delta = new_delta

    def train_batch(self, xs, ys, lr):
        """One SGD step on a mini-batch: average gradients, then update once."""
        gW = [[[0.0] * len(r) for r in W] for W in self.W]
        gb = [[0.0] * len(b) for b in self.b]
        loss = 0.0
        for x, y in zip(xs, ys):
            p, acts = self.forward(x, cache=True)
            loss += bce(p, y)
            self._backward(y, acts, gW, gb)
        n = len(xs)
        for layer in range(len(self.W)):
            for i in range(len(self.W[layer])):
                self.b[layer][i] -= lr * gb[layer][i] / n
                for j in range(len(self.W[layer][i])):
                    self.W[layer][i][j] -= lr * gW[layer][i][j] / n
        return loss / n

    def evaluate(self, X, y):
        loss = sum(bce(self.forward(x), t) for x, t in zip(X, y)) / len(X)
        acc = sum(self.predict(x) == t for x, t in zip(X, y)) / len(X)
        return loss, acc
