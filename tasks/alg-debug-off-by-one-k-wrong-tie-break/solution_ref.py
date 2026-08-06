import math
import numpy as np

def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    k: int
) -> np.ndarray:
    """Correct implementation of k‑Nearest Neighbours with smallest‑label tie‑break."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]

    preds = []
    for i in range(n_test):
        dists = []
        for j in range(n_train):
            sum_sq = 0.0
            for d in range(n_features):
                diff = X_train[j, d] - X_test[i, d]
                sum_sq += diff * diff
            dists.append((math.sqrt(sum_sq), j))

        sorted_dists = sorted(dists)

        counts = {}
        for idx in range(k):
            j = sorted_dists[idx][1]
            lbl = int(y_train[j])
            counts[lbl] = counts.get(lbl, 0) + 1

        best_label = None
        max_count = -1
        for lbl in sorted(counts.keys()):
            if counts[lbl] > max_count:
                max_count = counts[lbl]
                best_label = lbl

        preds.append(best_label)

    return np.array(preds)
