import sys; sys.path.append(".")
import numpy as np
from sklearn.naive_bayes import GaussianNB as SkGNB
from naive_bayes import GaussianNB
from data import get_splits

d = get_splits()
Xtr, ytr, Xva = d["X_train"], d["y_train"], d["X_val"]

mine = GaussianNB(var_smoothing=1e-9).fit(Xtr, ytr)
sk = SkGNB(var_smoothing=1e-9).fit(Xtr, ytr)

print("max |mean - sk.theta_|:", np.abs(mine.mean_ - sk.theta_).max())
print("max |var  - sk.var_|  :", np.abs(mine.var_ - sk.var_).max())
print("mean khớp (atol=1e-5):", np.allclose(mine.mean_, sk.theta_, atol=1e-5))
print("var khớp  (atol=1e-5):", np.allclose(mine.var_, sk.var_, atol=1e-5))

# So sánh công bằng hơn: cho sklearn cũng tính trên float64
sk64 = SkGNB(var_smoothing=1e-9).fit(Xtr.astype(np.float64), ytr)
print("mean khớp (float64):", np.allclose(mine.mean_, sk64.theta_))
print("var khớp  (float64):", np.allclose(mine.var_, sk64.var_))

agree = (mine.predict(Xva) == sk.predict(Xva)).mean()
print("tỉ lệ dự đoán giống sklearn:", agree)   