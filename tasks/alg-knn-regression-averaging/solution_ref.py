def knn_regression_average(X_train: list[list[float]],
                           y_train: list[float],
                           X_query: list[list[float]],
                           k: int) -> list[float]:
    n_train = len(X_train)
    if k > n_train:
        raise ValueError("k cannot exceed number of training samples")

    n_query = len(X_query)
    d = len(X_train[0]) if n_train > 0 else 0

    preds = [0.0] * n_query

    for i in range(n_query):
        dists = []
        for j in range(n_train):
            d_sq = 0.0
            for l in range(d):
                diff = X_train[j][l] - X_query[i][l]
                d_sq += diff * diff
            dists.append([d_sq, j])

        for p in range(k):
            min_idx = p
            for q in range(p + 1, n_train):
                if dists[q][0] < dists[min_idx][0]:
                    min_idx = q

            temp = dists[p]
            dists[p] = dists[min_idx]
            dists[min_idx] = temp

        val_sum = 0.0
        for p in range(k):
            val_sum += y_train[dists[p][1]]

        preds[i] = val_sum / k

    return preds
