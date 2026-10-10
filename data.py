import numpy as np
from pathlib import Path
import sklearn.datasets
CACHE = Path("data/mnist.npz")

def load_mnist_raw():
    """Trả về (X_train, y_train, X_test, y_test), X dạng uint8 (N, 784)."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X, y = X.astype(np.uint8), y.astype(np.int64)
    out = (X[:60000], y[:60000], X[60000:], y[60000:])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, X_train=out[0], y_train=out[1], X_test=out[2], y_test=out[3])
    return out

def get_splits(val_size=10000, seed=42):
    X_tr, y_tr, X_te, y_te = load_mnist_raw()
    X_tr = X_tr.astype(np.float32) / 255.0
    X_te = X_te.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_tr))
    va, tr = idx[:val_size], idx[val_size:]
    return {
        "X_train": X_tr[tr], "y_train": y_tr[tr],
        "X_val":   X_tr[va], "y_val":   y_tr[va],
        "X_test":  X_te,     "y_test":  y_te,
    }

def one_hot(y, k=10):
    Y = np.zeros((len(y), k), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y