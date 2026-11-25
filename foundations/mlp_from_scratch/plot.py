"""Plotting only: NumPy/Matplotlib are used here and nowhere else."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses, path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy"); plt.yscale("log")
    plt.title("Learning curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_decision_boundary(model, X, y, path, step=0.05):
    X_arr = np.array(X)
    xs = np.arange(X_arr[:, 0].min() - 0.5, X_arr[:, 0].max() + 0.5, step)
    ys = np.arange(X_arr[:, 1].min() - 0.5, X_arr[:, 1].max() + 0.5, step)
    Z = np.array([[model.forward([a, b]) for a in xs] for b in ys])
    plt.figure(figsize=(6, 4.5))
    plt.contourf(xs, ys, Z, levels=20, cmap="RdBu", alpha=0.7)
    plt.contour(xs, ys, Z, levels=[0.5], colors="k", linewidths=1.5)
    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y, cmap="RdBu", edgecolors="k", s=18)
    plt.title("Decision boundary (pure-Python MLP)")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
