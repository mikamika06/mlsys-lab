import numpy as np

def _ref_knn(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, k: int) -> np.ndarray:
    """Reference implementation of kNN with correct tie‑breaking."""
    # Compute pairwise Euclidean distances (test × train)
    dists = np.linalg.norm(X_train[None] - X_test[:, None], axis=2)
    # Indices of the k nearest neighbours for each test point
    idxs = np.argsort(dists, axis=1)[:, :k]
    preds = []
    max_label = int(y_train.max())
    for i in range(X_test.shape[0]):
        labels = y_train[idxs[i]]
        counts = np.bincount(labels, minlength=max_label + 1)
        # np.argmax returns the first index of the maximum value,
        # which is exactly the smallest label when there is a tie.
        preds.append(int(np.argmax(counts)))
    return np.array(preds)

def grade(sol, fx) -> dict:
    """Grade candidate implementation against reference."""
    rng = np.random.default_rng(0)
    ok = 1.0

    # 1. Random dataset
    for _ in range(3):
        m, d, n_test = 50, 5, 10
        X_train = rng.standard_normal((m, d))
        y_train = rng.integers(0, 5, size=m)
        X_test  = rng.standard_normal((n_test, d))
        k = 3
        try:
            cand = sol.predict_knn(X_train, y_train, X_test, k)
            ref = _ref_knn(X_train, y_train, X_test, k)
        except Exception:
            return {"argmax_agreement": 0.0}
        if not np.array_equal(cand, ref):
            ok = 0.0
            break

    # 2. Tie‑breaking adversarial case (smallest label rule)
    X_train = np.array([[0, 0], [1, 0]])
    y_train = np.array([0, 1])
    X_test  = np.array([[0.5, 0]])
    k = 2
    try:
        cand = sol.predict_knn(X_train, y_train, X_test, k)
        ref = _ref_knn(X_train, y_train, X_test, k)
    except Exception:
        return {"argmax_agreement": 0.0}
    if not np.array_equal(cand, ref):
        ok = 0.0

    # 3. Off‑by‑one adversarial case
    X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y_train = np.array([1, 1, 0, 0])
    X_test  = np.array([[0.5, 0.5]])
    k = 3
    try:
        cand = sol.predict_knn(X_train, y_train, X_test, k)
        ref = _ref_knn(X_train, y_train, X_test, k)
    except Exception:
        return {"argmax_agreement": 0.0}
    if not np.array_equal(cand, ref):
        ok = 0.0

    return {"argmax_agreement": ok}
