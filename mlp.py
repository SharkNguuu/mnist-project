import numpy as np
from data import one_hot


class MLP:
    def __init__(self, n_in=784, n_hidden=128, n_out=10, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.b1 = np.zeros(n_hidden, dtype=dtype)
        self.W2 = (rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)).astype(dtype)
        self.b2 = np.zeros(n_out, dtype=dtype)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ self.W2 + self.b2
        return Z1, A1, Z2

    def loss_and_grads(self, X, Y, l2=0.0):
        N = X.shape[0]
        Z1, A1, Z2 = self.forward(X)
        Zs = Z2 - Z2.max(axis=1, keepdims=True)                   # ổn định số
        log_P = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
        loss = -np.mean(np.sum(Y * log_P, axis=1))
        loss += 0.5 * l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        # lan truyền ngược
        dZ2 = (np.exp(log_P) - Y) / N
        dW2 = A1.T @ dZ2 + l2 * self.W2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1 + l2 * self.W1
        db1 = dZ1.sum(axis=0)
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, X, chunk=10000):
        out = [np.argmax(self.forward(X[s:s + chunk])[2], axis=1) for s in range(0, len(X), chunk)]
        return np.concatenate(out)


def eval_loss_acc(net, X, y, chunk=10000):
    """Cross-entropy trung bình (không tính L2) và accuracy, chia lô cho đỡ tốn RAM."""
    total, correct = 0.0, 0
    for s in range(0, len(X), chunk):
        Z2 = net.forward(X[s:s + chunk])[2]
        Zs = Z2 - Z2.max(axis=1, keepdims=True)
        log_P = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
        yb = y[s:s + chunk]
        total += -log_P[np.arange(len(yb)), yb].sum()
        correct += (np.argmax(Z2, axis=1) == yb).sum()
    return total / len(X), correct / len(X)


def train_mlp(net, X, y, X_val, y_val, lr=0.1, epochs=20, batch_size=128, l2=0.0, seed=0, track=True):
    rng = np.random.default_rng(seed)
    N = len(X)
    Y = one_hot(y)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        perm = rng.permutation(N)                       # xáo trộn mỗi epoch
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            _, grads = net.loss_and_grads(X[idx], Y[idx], l2)
            net.apply(grads, lr)
        if track:
            tl, _ = eval_loss_acc(net, X, y)
            vl, va = eval_loss_acc(net, X_val, y_val)
            hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
    return hist