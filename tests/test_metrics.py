import sys; sys.path.append(".")
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix as sk_cm
from metrics import evaluate, confusion_matrix

rng = np.random.default_rng(0)
yt = rng.integers(0, 10, 5000)
yp = np.where(rng.random(5000) < 0.7, yt, rng.integers(0, 10, 5000))

r = evaluate(yt, yp)
assert np.array_equal(confusion_matrix(yt, yp), sk_cm(yt, yp))
assert np.isclose(r["accuracy"], accuracy_score(yt, yp))
assert np.isclose(r["precision_macro"], precision_score(yt, yp, average="macro"))
assert np.isclose(r["recall_macro"], recall_score(yt, yp, average="macro"))
assert np.isclose(r["f1_macro"], f1_score(yt, yp, average="macro"))
print("metrics OK")