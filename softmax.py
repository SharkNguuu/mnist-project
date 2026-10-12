import numpy as np
from data import one_hot

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def loss_and_grad(W, b, X, Y, l2=0.0):
    """Cross-entropy trung bình (+ phạt L2 trên W) và gradient theo W, b."""
    N = X.shape[0]
    Z = X @ W + b
    Zs = Z - Z.max(axis=1, keepdims=True)
    log_P = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))   # log-softmax ổn định
    loss = -np.mean(np.sum(Y * log_P, axis=1)) + 0.5 * l2 * np.sum(W * W)
    dZ = (np.exp(log_P) - Y) / N
    dW = X.T @ dZ + l2 * W
    db = dZ.sum(axis=0)
    return loss, dW, db

def predict(W, b, X):
    return np.argmax(X @ W + b, axis=1)
def full_eval(W, b, X, y, Y, chunk=10000):
    """Loss (cross-entropy trung bình, không tính L2) và accuracy trên cả tập, chia lô cho đỡ tốn RAM."""
    total, correct = 0.0, 0
    for s in range(0, len(X), chunk):
        Z = X[s:s + chunk] @ W + b
        Zs = Z - Z.max(axis=1, keepdims=True)
        log_P = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
        total += -np.sum(Y[s:s + chunk] * log_P)
        correct += np.sum(np.argmax(Z, axis=1) == y[s:s + chunk])
    return total / len(X), correct / len(X)


def train(X, y, X_val, y_val, lr=0.5, epochs=30, batch_size=128, l2=0.0, seed=0, track=True):
    rng = np.random.default_rng(seed)
    N, d = X.shape
    Y, Y_val = one_hot(y), one_hot(y_val)
    W = np.zeros((d, 10), dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        perm = rng.permutation(N)                      # xáo trộn mỗi epoch
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            _, dW, db = loss_and_grad(W, b, X[idx], Y[idx], l2)
            W -= lr * dW
            b -= lr * db
        if track:
            tl, _ = full_eval(W, b, X, y, Y)
            vl, va = full_eval(W, b, X_val, y_val, Y_val)
            hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
    return W, b, hist