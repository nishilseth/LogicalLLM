import random
from sklearn.datasets import make_moons  # only used to generate data


def load_data(n_samples=400, noise=0.2, test_frac=0.2, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    data = [([float(a), float(b)], int(t)) for (a, b), t in zip(X, y)]
    random.Random(seed).shuffle(data)
    split = int((1 - test_frac) * len(data))
    train, test = data[:split], data[split:]
    return ([d[0] for d in train], [d[1] for d in train],
            [d[0] for d in test], [d[1] for d in test])
