import numpy as np


def _oracle(X, threshold):
    absmax = np.max(np.abs(X), axis=0)
    idx = np.nonzero(absmax >= threshold)[0]
    return np.sort(idx).astype(np.int64)


def grade(sol, fx) -> dict:
    """
    Builds random token x hidden activation matrices with a handful of
    injected large-magnitude outlier columns, computes the per-column
    absmax threshold set with a NumPy oracle, and checks the submission's
    returned index set is exactly identical.
    """
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(8):
        try:
            n = int(rng.integers(5, 30))
            d = int(rng.integers(4, 20))
            threshold = 6.0
            X = (rng.normal(size=(n, d)) * rng.uniform(0.1, 1.5)).astype(np.float64)

            n_outliers = int(rng.integers(0, min(4, d) + 1))
            outlier_cols = rng.choice(d, size=n_outliers, replace=False)
            for c in outlier_cols:
                row = int(rng.integers(0, n))
                sign = rng.choice([-1.0, 1.0])
                X[row, c] = sign * rng.uniform(6.0, 20.0)

            expected = _oracle(X, threshold)
            got = sol.detect_outlier_columns(X.copy(), threshold)
            got = np.asarray(got, dtype=np.int64).ravel()
        except Exception:
            ok = 0.0
            break

        if set(got.tolist()) != set(expected.tolist()) or len(got) != len(expected):
            ok = 0.0
            break

    return {"exact_match": ok}
