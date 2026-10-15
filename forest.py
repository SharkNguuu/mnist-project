import numpy as np


class DecisionTree:
    """Cây quyết định phân lớp trên đặc trưng nhị phân (0/1), tiêu chí Gini.
    Xb phải là mảng float32 chỉ gồm 0 và 1."""

    def __init__(self, max_depth=12, min_samples_leaf=1, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features      # None = xét mọi pixel; số nguyên = chọn ngẫu nhiên mỗi nút
        self.seed = seed

    def _best_split(self, Xn, Yn, feats):
        """Trả về (pixel tốt nhất, Gini sau tách) hoặc None nếu không có cách tách hợp lệ."""
        n = Xn.shape[0]
        Xs = Xn if feats is None else Xn[:, feats]
        R = (Xs.T @ Yn).astype(np.float64)               # (f, K): đếm lớp ở nhánh pixel = 1
        T = Yn.sum(axis=0, dtype=np.float64)             # (K,): đếm lớp của cả nút
        L = T[None, :] - R                               # đếm lớp ở nhánh pixel = 0
        nR = R.sum(axis=1)
        nL = n - nR
        valid = (nL >= self.min_samples_leaf) & (nR >= self.min_samples_leaf)
        if not valid.any():
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            wg = (nL - (L ** 2).sum(axis=1) / nL + nR - (R ** 2).sum(axis=1) / nR) / n
        wg = np.where(valid, wg, np.inf)
        j = int(np.argmin(wg))
        return (j if feats is None else int(feats[j])), float(wg[j])

    def fit(self, Xb, y, n_classes=10):
        y = np.asarray(y)
        N, d = Xb.shape
        K = n_classes
        mf = d if self.max_features is None else int(self.max_features)
        rng = np.random.default_rng(self.seed)
        Y = np.zeros((N, K), dtype=np.float32)
        Y[np.arange(N), y] = 1.0
        feature, left, right, value = [], [], [], []

        def new_node(idx):
            feature.append(-1); left.append(-1); right.append(-1)
            value.append(Y[idx].sum(axis=0, dtype=np.float64))
            return len(feature) - 1

        all_idx = np.arange(N)
        stack = [(new_node(all_idx), all_idx, 0)]
        while stack:
            node, idx, depth = stack.pop()
            counts = value[node]
            n = len(idx)
            if depth >= self.max_depth or n < 2 * self.min_samples_leaf or counts.max() == n:
                continue                                  # thành lá: hết độ sâu, quá ít mẫu, hoặc đã thuần
            feats = None if mf >= d else rng.choice(d, mf, replace=False)
            Xn = Xb[idx]
            res = self._best_split(Xn, Y[idx], feats)
            if res is None:
                continue
            j, wg = res
            parent_gini = 1.0 - np.sum((counts / n) ** 2)
            if wg >= parent_gini - 1e-12:
                continue                                  # không giảm được độ tạp
            go_right = Xn[:, j] > 0
            idx_r, idx_l = idx[go_right], idx[~go_right]
            feature[node] = j
            left[node] = new_node(idx_l)
            right[node] = new_node(idx_r)
            stack.append((left[node], idx_l, depth + 1))
            stack.append((right[node], idx_r, depth + 1))

        self.feature_ = np.array(feature)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)                     # (số nút, K): đếm lớp tại mỗi nút
        self.pred_ = self.value_.argmax(axis=1)
        return self

    def apply(self, Xb):
        """Trả về chỉ số nút lá mà mỗi mẫu rơi vào (duyệt theo từng tầng cho cả lô)."""
        rows = np.arange(len(Xb))
        node = np.zeros(len(Xb), dtype=np.int64)
        for _ in range(self.max_depth + 1):
            f = self.feature_[node]
            internal = f >= 0
            if not internal.any():
                break
            go_right = Xb[rows, np.where(internal, f, 0)] > 0
            nxt = np.where(go_right, self.right_[node], self.left_[node])
            node = np.where(internal, nxt, node)
        return node

    def predict_proba(self, Xb):
        v = self.value_[self.apply(Xb)]
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, Xb):
        return self.pred_[self.apply(Xb)]