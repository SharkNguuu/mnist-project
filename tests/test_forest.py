import sys; sys.path.append(".")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data import get_splits
from forest import DecisionTree, RandomForest

d = get_splits()
Xtr = (d["X_train"] > 0.5).astype(np.float32); ytr = d["y_train"]
Xva = (d["X_val"] > 0.5).astype(np.float32);   yva = d["y_val"]

# 1) Rừng 1 cây, không bootstrap, xét mọi pixel  ==  DecisionTree
f1 = RandomForest(n_trees=1, max_depth=12, max_features=None, bootstrap=False).fit(Xtr, ytr)
t1 = DecisionTree(max_depth=12).fit(Xtr, ytr)
print("1 cây == DecisionTree, dự đoán trùng:", (f1.predict(Xva) == t1.predict(Xva)).mean())

# 2) Xác suất của rừng cộng bằng 1
f5 = RandomForest(n_trees=5, max_depth=12).fit(Xtr, ytr)
print("tổng xác suất bằng 1:", np.allclose(f5.predict_proba(Xva).sum(axis=1), 1.0))

# 3) So với sklearn (cùng độ sâu, số cây, 28 pixel mỗi nút, cùng dữ liệu nhị phân)
mine = RandomForest(n_trees=30, max_depth=12).fit(Xtr, ytr)
sk = RandomForestClassifier(n_estimators=30, max_depth=12, max_features=28,
                            random_state=0, n_jobs=-1).fit(Xtr, ytr)
pm, ps = mine.predict(Xva), sk.predict(Xva)
print(f"val acc mình {(pm == yva).mean()*100:.2f}% | sklearn {(ps == yva).mean()*100:.2f}% "
      f"| dự đoán trùng {(pm == ps).mean()*100:.2f}%")