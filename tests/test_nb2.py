import sys; sys.path.append(".")
import numpy as np
from sklearn.naive_bayes import MultinomialNB as SkMNB, BernoulliNB as SkBNB
from naive_bayes import MultinomialNB, BernoulliNB
from data import get_splits

d = get_splits()
Xtr, ytr, Xva = d["X_train"].astype(np.float64), d["y_train"], d["X_val"].astype(np.float64)

# Multinomial
m, s = MultinomialNB(alpha=1.0).fit(Xtr, ytr), SkMNB(alpha=1.0).fit(Xtr, ytr)
print("MNB  max|feature_log_prob|:", np.abs(m.feature_log_prob_ - s.feature_log_prob_).max())
print("MNB  log prior khớp:", np.allclose(m.log_prior_, s.class_log_prior_))
print("MNB  dự đoán giống sklearn:", (m.predict(Xva) == s.predict(Xva)).mean())

# Bernoulli (sklearn có sẵn tham số binarize)
b, sb = BernoulliNB(alpha=1.0, binarize=0.5).fit(Xtr, ytr), SkBNB(alpha=1.0, binarize=0.5).fit(Xtr, ytr)
print("BNB  max|feature_log_prob|:", np.abs(b.feature_log_prob_ - sb.feature_log_prob_).max())
print("BNB  log prior khớp:", np.allclose(b.log_prior_, sb.class_log_prior_))
print("BNB  dự đoán giống sklearn:", (b.predict(Xva) == sb.predict(Xva)).mean())