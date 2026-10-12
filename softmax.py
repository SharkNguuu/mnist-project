import numpy as np

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