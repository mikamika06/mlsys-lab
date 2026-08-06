import numpy as np
import math

def knn_vote_divergence(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test:  np.ndarray,
                        k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Return uniform and distance‑weighted kNN predictions for each point in X_test.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_test.shape[1] if n_test > 0 else 0

    max_y = -1
    for i in range(n_train):
        val = int(y_train[i])
        if val > max_y:
            max_y = val
    n_classes = max_y + 1

    out_uniform = np.zeros(n_test, dtype=np.int64)
    out_weighted = np.zeros(n_test, dtype=np.int64)
    eps = 1e-12

    for i in range(n_test):
        dists_i = []
        for j in range(n_train):
            d_sq = 0.0
            for f in range(n_features):
                diff = X_test[i][f] - X_train[j][f]
                d_sq += diff * diff
            dists_i.append(math.sqrt(d_sq))

        k_nearest_idx = []
        used = [False] * n_train
        for _ in range(k):
            min_d = float('inf')
            min_idx = -1
            for j in range(n_train):
                if not used[j]:
                    if dists_i[j] < min_d:
                        min_d = dists_i[j]
                        min_idx = j
            if min_idx != -1:
                used[min_idx] = True
                k_nearest_idx.append(min_idx)

        counts = [0] * n_classes
        for j in k_nearest_idx:
            cls = int(y_train[j])
            counts[cls] += 1

        best_count = -1
        best_cls_uni = 0
        for c in range(n_classes):
            if counts[c] > best_count:
                best_count = counts[c]
                best_cls_uni = c
        out_uniform[i] = best_cls_uni

        weighted_sums = [0.0] * n_classes
        for c in range(n_classes):
            s = 0.0
            for j in k_nearest_idx:
                if int(y_train[j]) == c:
                    s += 1.0 / (dists_i[j] + eps)
                else:
                    s += 0.0
            weighted_sums[c] = s

        best_weight = -1.0
        best_cls_w = 0
        for c in range(n_classes):
            if weighted_sums[c] > best_weight:
                best_weight = weighted_sums[c]
                best_cls_w = c
        out_weighted[i] = best_cls_w

    return out_uniform, out_weighted
