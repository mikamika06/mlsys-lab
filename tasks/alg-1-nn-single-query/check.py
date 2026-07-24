import numpy as np

def _oracle_knn_predict(X_tr, y_tr, q):
    # Vectorised Euclidean distance computation
    dists = np.linalg.norm(X_tr - q, axis=1)
    idx = int(np.argmin(dists))
    return int(y_tr[idx])

def grade(sol, fx) -> dict:
    """
    The grader creates several random datasets and checks that the user’s
    implementation returns exactly the same label as the NumPy oracle.
    """
    np.random.seed(0)
    ok = 1.0
    for _ in range(5):
        try:
            n_samples = np.random.randint(3, 20)
            n_features = np.random.randint(2, 10)
            X_tr = np.random.randn(n_samples, n_features).astype(np.float64)
            y_tr = np.random.randint(-1000, 1000, size=n_samples)
            q = np.random.randn(n_features).astype(np.float64)

            expected = _oracle_knn_predict(X_tr, y_tr, q)
            got = sol.knn_predict(X_tr, y_tr, q)
        except Exception:
            ok = 0.0
            break

        if int(got) != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
