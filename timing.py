import json, os, platform, sys, time
import numpy as np
from pathlib import Path
from data import get_splits
from final_models import MODELS

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
d = get_splits()
X_val = d["X_val"]

# Khởi động (không tính giờ): tránh chi phí gọi lần đầu của thư viện
small = {k: v[:2000] for k, v in d.items()}
for name, fit in MODELS.items():
    fit(small)(X_val[:1000])
print("xong khởi động", flush=True)

ROUNDS = 3
fit_t = {n: [] for n in MODELS}
pred_t = {n: [] for n in MODELS}
for r in range(ROUNDS):
    for name, fit in MODELS.items():
        t0 = time.perf_counter(); predict = fit(d); fit_t[name].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); predict(X_val);   pred_t[name].append(time.perf_counter() - t0)
        print(f"vòng {r+1} | {name:14s} | fit {fit_t[name][-1]:7.2f}s | predict(10.000 ảnh) {pred_t[name][-1]:.3f}s", flush=True)

info = dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
            python=sys.version.split()[0], numpy=np.__version__)
with open(OUT / "timing.json", "w") as f:
    json.dump(dict(fit=fit_t, predict=pred_t, env=info), f)
print(info)