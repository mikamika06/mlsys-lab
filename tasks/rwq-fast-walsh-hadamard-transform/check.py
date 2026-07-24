import numpy as np

from mlsys import scorers


def _hadamard_matrix(n: int) -> np.ndarray:
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def _oracle_fwht(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    H = _hadamard_matrix(n)
    return (H @ x) / np.sqrt(n)


def _build_cases():
    cases = []
    for seed, n in [(0, 8), (1, 32), (2, 128), (3, 256)]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        cases.append(x)
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for x in _build_cases():
        ref = _oracle_fwht(x)
        try:
            got = sol.fwht(x.copy())
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(ref, got)
        if not np.isfinite(err):
            err = float("inf")
        worst = max(worst, err)

    return {"max_abs_err": worst}
