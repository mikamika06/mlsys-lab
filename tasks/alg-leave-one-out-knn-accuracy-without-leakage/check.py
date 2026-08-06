import numpy as np


def _oracle_loo_knn_predict(X, y, k, n_classes):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    diff = X[:, None, :] - X[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    np.fill_diagonal(distances, np.inf)

    neighbors = np.argsort(distances, axis=1)[:, :k]
    preds = []
    for row in neighbors:
        counts = np.bincount(y[row], minlength=n_classes)
        preds.append(int(np.argmax(counts)))
    return preds


def _argmax_agreement(a, b):
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def grade(sol, fx) -> dict:
    cases = [
        (
            [[0.0], [0.05], [1.0], [1.05]],
            [0, 0, 1, 1],
            1,
            2,
        ),
        (
            [[0.0], [0.0], [0.1], [2.0], [2.1]],
            [1, 0, 0, 1, 1],
            2,
            2,
        ),
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]],
            [0, 0, 0, 1, 1, 1],
            3,
            2,
        ),
    ]

    scores = []
    for X, y, k, n_classes in cases:
        ref = _oracle_loo_knn_predict(X, y, k, n_classes)
        try:
            got = sol.loo_knn_predict(X, y, k, n_classes)
        except Exception:
            return {"argmax_agreement": 0.0}
        scores.append(_argmax_agreement(got, ref))

    return {"argmax_agreement": float(np.mean(scores))}
