import numpy as np
from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import softmax as sm
from mlp import MLP, train_mlp
from forest import RandomForest


def _binarize(X):
    return (X > 0.5).astype(np.float32)


def fit_gnb(d):
    m = GaussianNB(var_smoothing=0.1).fit(d["X_train"], d["y_train"])
    return m.predict


def fit_mnb(d):
    m = MultinomialNB(alpha=1e-4).fit(d["X_train"], d["y_train"])
    return m.predict


def fit_bnb(d):
    m = BernoulliNB(alpha=1e-3, binarize=0.5).fit(d["X_train"], d["y_train"])
    return m.predict


def fit_softmax(d):
    W, b, _ = sm.train(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                       lr=0.1, epochs=30, batch_size=128, seed=0, track=False)
    return lambda X: sm.predict(W, b, X)


def fit_mlp(d):
    net = MLP(n_hidden=512, seed=0)
    train_mlp(net, d["X_train"], d["y_train"], d["X_val"], d["y_val"],
              lr=0.3, epochs=20, batch_size=128, seed=0, track=False)
    return net.predict


def fit_rf(d):
    f = RandomForest(n_trees=61, max_depth=20, seed=1).fit(_binarize(d["X_train"]), d["y_train"])
    return lambda X: f.predict(_binarize(X))


MODELS = {
    "GaussianNB": fit_gnb,
    "MultinomialNB": fit_mnb,
    "BernoulliNB": fit_bnb,
    "Softmax": fit_softmax,
    "MLP": fit_mlp,
    "RandomForest": fit_rf,
}