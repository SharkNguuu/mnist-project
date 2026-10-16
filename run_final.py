import sys, json
import numpy as np
from pathlib import Path
from data import get_splits
from metrics import evaluate
from final_models import MODELS

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
if (OUT / "metrics.json").exists() and "--force" not in sys.argv:
    sys.exit("results/metrics.json đã tồn tại: tập test đã được dùng. Không chạy lại.")

# Accuracy validation đã biết từ các thí nghiệm trước (để kiểm tra tái lập)
KNOWN_VAL = {"GaussianNB": 79.78, "MultinomialNB": 82.38, "BernoulliNB": 83.53,
             "Softmax": 92.02, "MLP": 97.86, "RandomForest": 96.15}

d = get_splits()
results = {}
for name, fit in MODELS.items():
    print(f"\n=== {name} ===", flush=True)
    predict = fit(d)
    preds = {s: predict(d[f"X_{s}"]) for s in ["train", "val", "test"]}
    results[name] = {}
    for s, p in preds.items():
        r = evaluate(d[f"y_{s}"], p)
        results[name][s] = {k: float(v) for k, v in r.items() if k != "confusion_matrix"}
        results[name][s]["confusion_matrix"] = r["confusion_matrix"].tolist()
        print(f"  {s:5s}: acc {r['accuracy']*100:.2f}% | P {r['precision_macro']*100:.2f}% | "
              f"R {r['recall_macro']*100:.2f}% | F1 {r['f1_macro']*100:.2f}%")
    np.savez_compressed(OUT / f"preds_{name}.npz", **preds)
    v = results[name]["val"]["accuracy"] * 100
    print(f"  [kiểm tra] val {v:.2f}% so với đã biết {KNOWN_VAL[name]:.2f}% (lệch {v - KNOWN_VAL[name]:+.2f})")

with open(OUT / "metrics.json", "w") as f:
    json.dump(results, f)
print("\nĐã lưu results/metrics.json và preds_*.npz")