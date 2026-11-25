import argparse, os, random
from mlp import MLP
from dataset import load_data
from plot import plot_loss, plot_decision_boundary

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=5, help="neurons in the hidden layer (3-5 works)")
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--out", default="results")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
X_tr, y_tr, X_te, y_te = load_data()
model = MLP([2, args.hidden, 1], seed=0)
rng = random.Random(0)
idx = list(range(len(X_tr)))
tr_hist, te_hist = [], []

for epoch in range(1, args.epochs + 1):
    rng.shuffle(idx)
    for i in range(0, len(idx), args.batch_size):
        b = idx[i:i + args.batch_size]
        model.train_batch([X_tr[j] for j in b], [y_tr[j] for j in b], args.lr)
    tr_loss, tr_acc = model.evaluate(X_tr, y_tr)
    te_loss, te_acc = model.evaluate(X_te, y_te)
    tr_hist.append(tr_loss); te_hist.append(te_loss)
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch {epoch:4d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | test loss {te_loss:.4f} acc {te_acc:.3f}")

plot_loss(tr_hist, te_hist, f"{args.out}/learning_curve.png")
plot_decision_boundary(model, X_te, y_te, f"{args.out}/decision_boundary.png")
print(f"final test accuracy: {te_acc:.3f}  (hidden={args.hidden})")
