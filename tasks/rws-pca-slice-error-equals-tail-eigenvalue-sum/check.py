import numpy as np


def _oracle_tail_sum(X: np.ndarray, k: int) -> float:
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    G = X.T @ X
    eigvals_asc, _ = np.linalg.eigh(G)  # ascending order
    # tail = the (d-k) smallest eigenvalues, i.e. everything dropped by
    # keeping only the top-k
    tail = eigvals_asc[: d - k] if k < d else np.array([])
    return float(np.sum(tail))


def _cases(rng: np.random.Generator):
    cases = []

    # hand-checkable 2x2
    cases.append((np.array([[3.0, 0.0], [0.0, 1.0]]), 1))
    cases.append((np.array([[3.0, 0.0], [0.0, 1.0]]), 0))
    cases.append((np.array([[3.0, 0.0], [0.0, 1.0]]), 2))

    # random matrices, several k values including edges
    n, d = 40, 7
    X = rng.standard_normal((n, d)) @ np.diag(rng.uniform(0.1, 5.0, size=d))
    for k in [0, 1, 3, 6, 7]:
        cases.append((X, k))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    all_cases = [(fx["pca_x"], int(fx["pca_k"]))] + _cases(rng)

    worst = 0.0
    for X, k in all_cases:
        expected = _oracle_tail_sum(X, k)
        try:
            got = float(sol.pca_slice_error(np.array(X, copy=True), int(k)))
        except Exception:
            return {"max_abs_err": float("inf")}

        worst = max(worst, abs(got - expected))

    return {"max_abs_err": worst}
