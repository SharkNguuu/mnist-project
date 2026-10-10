import numpy as np
import time

def confusion_matrix(y_true, y_pred, k=10):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)

def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

def evaluate(y_true, y_pred, k=10):
    C = confusion_matrix(y_true, y_pred, k)
    tp = np.diag(C).astype(float)
    precision = _div(tp, C.sum(axis=0))
    recall    = _div(tp, C.sum(axis=1))
    f1        = _div(2 * precision * recall, precision + recall)
    return {
        "accuracy": tp.sum() / C.sum(),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "confusion_matrix": C,
    }

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0