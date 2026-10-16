import sys, json
from pathlib import Path
from data import get_splits
from metrics import evaluate
from final_models import MODELS

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
if (OUT / "fashion_metrics.json").exists() and "--force" not in sys.argv:
    sys.exit("results/fashion_metrics.json đã tồn tại: tập test Fashion-MNIST đã được dùng.")

d = get_splits(name="fashion")
results = {}
for name, fit in MODELS.items():
    print(f"\n=== {name} ===", flush=True)
    predict = fit(d)
    results[name] = {}
    for s in ["train", "val", "test"]:
        r = evaluate(d[f"y_{s}"], predict(d[f"X_{s}"]))
        results[name][s] = {k: float(v) for k, v in r.items() if k != "confusion_matrix"}
        results[name][s]["confusion_matrix"] = r["confusion_matrix"].tolist()
        print(f"  {s:5s}: acc {r['accuracy']*100:.2f}% | P {r['precision_macro']*100:.2f}% | "
              f"R {r['recall_macro']*100:.2f}% | F1 {r['f1_macro']*100:.2f}%", flush=True)

with open(OUT / "fashion_metrics.json", "w") as f:
    json.dump(results, f)
print("\nĐã lưu results/fashion_metrics.json")