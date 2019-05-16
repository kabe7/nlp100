import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import scipy.sparse as spar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

k = 5  # 分割数
features = spar.load_npz('../data/features.npz')
polarity = np.load('../data/polarity.npy')
n = len(polarity)

partition = [int(m * n / k) for m in range(k + 1)]
section = [(start, end) for start, end in zip(partition[:k], partition[1:])]

tests = [
    ('precision', precision_score, []),
    ('recall', recall_score, []),
]

# 文章がpositiveであると予測される確率を計算する
probability = np.empty(n)
for start, end in section:
    fancy_idx = np.array([True if start <= i < end else False for i in range(n)], dtype=bool)
    # 検証用データ(validation)
    va_por, va_fs = polarity[fancy_idx], features[fancy_idx]
    # 学習用データ(training)
    tr_por, tr_fs = polarity[np.logical_not(fancy_idx)], features[np.logical_not(fancy_idx)]

    model = LogisticRegression(solver='lbfgs')
    model.fit(tr_fs, tr_por)
    probability[start:end] = model.predict_proba(va_fs)[:, 1]

# 確率が閾値以上ならpositiveと判定する
for threshold in np.linspace(0, max(probability), 100):
    for name, test, results in tests:
        predict = np.where(probability >= threshold, 1, -1)
        results.append(test(polarity, predict))

precision, recall = [results for _, _, results in tests]
plt.plot(recall, precision)
plt.title('precision-recall curve')
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()