import sys; sys.path.append(".")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from data import get_splits
from forest import DecisionTree

d = get_splits()
Xtr = (d["X_train"] > 0.5).astype(np.float32); ytr = d["y_train"]
Xva = (d["X_val"] > 0.5).astype(np.float32);   yva = d["y_val"]

# 1) Cây một tầng: pixel gốc và dự đoán phải trùng sklearn
mine = DecisionTree(max_depth=1).fit(Xtr, ytr)
sk = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=0).fit(Xtr, ytr)
print("pixel gốc: mình", mine.feature_[0], "| sklearn", sk.tree_.feature[0])
print("stump: dự đoán trùng sklearn:", (mine.predict(Xva) == sk.predict(Xva)).mean())

# 2) Cây độ sâu 12: so accuracy và độ trùng dự đoán
for dep in [6, 12]:
    mine = DecisionTree(max_depth=dep).fit(Xtr, ytr)
    sk = DecisionTreeClassifier(max_depth=dep, criterion="gini", random_state=0).fit(Xtr, ytr)
    pm, ps = mine.predict(Xva), sk.predict(Xva)
    print(f"độ sâu {dep}: val acc mình {(pm == yva).mean()*100:.2f}% | sklearn {(ps == yva).mean()*100:.2f}% "
          f"| dự đoán trùng {(pm == ps).mean()*100:.2f}%")