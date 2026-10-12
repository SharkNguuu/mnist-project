import numpy as np

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.class_prior_ = np.bincount(y) / len(y)
        self.classes_ = np.unique(y)
        k, d = len(self.classes_), X.shape[1]
        self.mean_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        self.log_prior_ = np.zeros(k)
        # cách làm của sklearn: ε = var_smoothing × phương sai lớn nhất trên toàn tập
        eps = self.var_smoothing * X.var(axis=0, dtype=np.float64).max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c].astype(np.float64)
            self.mean_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + eps
            self.log_prior_[i] = np.log(len(Xc) / len(X))
        return self

    def joint_log_likelihood(self, X, batch=5000):
        out = np.empty((len(X), len(self.classes_)))
        for s in range(0, len(X), batch):          # chia lô để không tốn RAM
            Xb = X[s:s + batch].astype(np.float64)
            for i in range(len(self.classes_)):
                term1 = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i]))
                term2 = -0.5 * np.sum((Xb - self.mean_[i]) ** 2 / self.var_[i], axis=1)
                out[s:s + batch, i] = self.log_prior_[i] + term1 + term2
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        k, d = len(self.classes_), X.shape[1]
        Y = np.zeros((len(y), k))
        Y[np.arange(len(y)), np.searchsorted(self.classes_, y)] = 1.0
        counts = Y.T @ X.astype(np.float64)                     # (k, d): tổng giá trị pixel theo lớp
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.log_prior_ = np.log(Y.sum(axis=0) / len(y))
        return self

    def joint_log_likelihood(self, X):
        return X.astype(np.float64) @ self.feature_log_prob_.T + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


class BernoulliNB:
    def __init__(self, alpha=1.0, binarize=0.5):
        self.alpha = alpha
        self.binarize = binarize

    def _bin(self, X):
        return (X > self.binarize).astype(np.float64)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        Y = np.zeros((len(y), k))
        Y[np.arange(len(y)), np.searchsorted(self.classes_, y)] = 1.0
        n_c = Y.sum(axis=0)                                      # số mẫu mỗi lớp
        n_ci = Y.T @ self._bin(X)                                # số mẫu lớp c có pixel i sáng
        p = (n_ci + self.alpha) / (n_c[:, None] + 2 * self.alpha)
        self.feature_log_prob_ = np.log(p)
        self.neg_log_prob_ = np.log(1 - p)
        self.log_prior_ = np.log(n_c / len(y))
        return self

    def joint_log_likelihood(self, X):
        Xb = self._bin(X)
        # Σ x·log p + (1-x)·log(1-p)  =  x @ (log p − log(1-p)).T + Σ log(1-p)
        delta = self.feature_log_prob_ - self.neg_log_prob_
        return Xb @ delta.T + self.neg_log_prob_.sum(axis=1) + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]