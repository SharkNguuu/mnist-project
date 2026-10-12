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
        for s in range(0, len(X), batch):        
            Xb = X[s:s + batch].astype(np.float64)
            for i in range(len(self.classes_)):
                term1 = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i]))
                term2 = -0.5 * np.sum((Xb - self.mean_[i]) ** 2 / self.var_[i], axis=1)
                out[s:s + batch, i] = self.log_prior_[i] + term1 + term2
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]

