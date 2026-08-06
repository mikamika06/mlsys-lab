import numpy as np

def knn_grid_labels(
    train_points: np.ndarray,
    train_labels: np.ndarray,
    grid_points: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """
    Vectorised k‑Nearest Neighbours classification on a dense grid.

    Parameters
    ----------
    train_points : (N, d) array of training samples.
    train_labels : (N,) integer labels in {0,…,C-1}.
    grid_points  : (M, d) query points to classify.
    k            : number of neighbours to consider.

    Returns
    -------
    logits : (M, C) one‑hot encoded predictions.
    """
    train_points = np.asarray(train_points, dtype=np.float64)
    grid_points  = np.asarray(grid_points,  dtype=np.float64)

    N, d = train_points.shape
    M     = grid_points.shape[0]
    C     = int(max(train_labels)) + 1

    logits_list = []
    for i in range(M):
        row_dists = []
        for j in range(N):
            s = 0.0
            for l in range(d):
                diff = grid_points[i, l] - train_points[j, l]
                s += diff * diff
            row_dists.append(s)

        indexed_dists = sorted(enumerate(row_dists), key=lambda x: x[1])
        idx = [indexed_dists[t][0] for t in range(k)]

        counts_row = [0] * C
        for t in range(k):
            label = int(train_labels[idx[t]])
            counts_row[label] += 1

        max_count = -1
        best_label = 0
        for label in range(C):
            if counts_row[label] > max_count:
                max_count = counts_row[label]
                best_label = label

        row_logit = [0.0] * C
        row_logit[best_label] = 1.0
        logits_list.append(row_logit)

    logits = np.asarray(logits_list, dtype=np.float64)
    return logits
