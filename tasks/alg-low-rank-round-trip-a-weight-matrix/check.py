import numpy as np

from mlsys import scorers


def _cases():
    """Deterministic (W, k) cases: square, wide, tall, decaying spectrum, full rank."""
    rng = np.random.default_rng(0)
    out = []

    # square, generic spectrum
    W = rng.standard_normal((10, 10))
    out.append((W, 3))

    # wide (m < n)
    W = rng.standard_normal((5, 12))
    out.append((W, 2))

    # tall (m > n)
    W = rng.standard_normal((15, 8))
    out.append((W, 4))

    # fast-decaying spectrum: low-rank plus small noise
    U = rng.standard_normal((14, 3))
    V = rng.standard_normal((3, 9))
    W = U @ V + 1e-3 * rng.standard_normal((14, 9))
    out.append((W, 3))

    # k == min(m, n): lossless round trip
    W = rng.standard_normal((6, 6))
    out.append((W, 6))

    # rank-deficient integer matrix
    W = np.arange(1.0, 13.0).reshape(4, 3)
    out.append((W, 1))

    return out


def _oracle(W, k):
    W = np.asarray(W, dtype=np.float64)
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def grade(sol, fx) -> dict:
    worst_recon = 0.0
    worst_factor = 0.0
    shape_ok = 1.0

    for W, k in _cases():
        m, n = W.shape
        ref = _oracle(W, k)

        try:
            got = np.asarray(sol.low_rank_reconstruct(W.copy(), k), dtype=np.float64)
        except Exception:
            return {
                "max_abs_err": float("inf"),
                "factor_max_abs_err": float("inf"),
                "shape_ok": 0.0,
            }
        if got.shape != ref.shape:
            return {
                "max_abs_err": float("inf"),
                "factor_max_abs_err": float("inf"),
                "shape_ok": 0.0,
            }
        worst_recon = max(worst_recon, scorers.max_abs_err(ref, got))

        try:
            pair = sol.low_rank_factors(W.copy(), k)
            A = np.asarray(pair[0], dtype=np.float64)
            B = np.asarray(pair[1], dtype=np.float64)
        except Exception:
            return {
                "max_abs_err": worst_recon,
                "factor_max_abs_err": float("inf"),
                "shape_ok": 0.0,
            }

        if A.shape != (m, k) or B.shape != (k, n):
            shape_ok = 0.0
            continue

        worst_factor = max(worst_factor, scorers.max_abs_err(ref, A @ B))

    if shape_ok == 0.0:
        worst_factor = float("inf")

    return {
        "max_abs_err": float(worst_recon),
        "factor_max_abs_err": float(worst_factor),
        "shape_ok": float(shape_ok),
    }
