import numpy as np
from mlsys import scorers


def _oracle_factors(A, k):
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def _oracle_reconstruct(U, S, Vt):
    return (U * S) @ Vt


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)

    cases = [
        (np.outer(np.linspace(1.0, 3.0, 80), np.linspace(2.0, 5.0, 100)), 1),
        (
            np.outer(np.arange(1, 40, dtype=np.float64), np.linspace(1.0, 2.0, 60))
            + np.outer(np.linspace(2.0, 4.0, 39), np.linspace(3.0, 4.0, 60)),
            2,
        ),
        (
            sum(
                np.outer(rng.normal(size=120), rng.normal(size=140))
                for _ in range(3)
            ),
            3,
        ),
    ]

    size_scores = []
    mse_scores = []

    for A, k in cases:
        try:
            U, S, Vt = sol.compress_svd(A, k)
            A_hat = sol.reconstruct_svd(U, S, Vt)
        except Exception:
            return {"size_ratio": 0.0, "mse": float("inf")}

        ref_U, ref_S, ref_Vt = _oracle_factors(A, k)
        ref_A = _oracle_reconstruct(ref_U, ref_S, ref_Vt)

        if np.asarray(U).shape != ref_U.shape:
            return {"size_ratio": 0.0, "mse": float("inf")}
        if np.asarray(S).shape != ref_S.shape:
            return {"size_ratio": 0.0, "mse": float("inf")}
        if np.asarray(Vt).shape != ref_Vt.shape:
            return {"size_ratio": 0.0, "mse": float("inf")}

        factors_bytes = (
            np.asarray(U).nbytes
            + np.asarray(S).nbytes
            + np.asarray(Vt).nbytes
        )
        if factors_bytes == 0:
            return {"size_ratio": 0.0, "mse": float("inf")}

        size_scores.append(float(A.nbytes / factors_bytes))
        mse_scores.append(scorers.mse(A, A_hat))

        if not np.allclose(A_hat, ref_A, atol=1e-8, rtol=1e-8):
            return {"size_ratio": 0.0, "mse": float("inf")}

    return {
        "size_ratio": float(min(size_scores)),
        "mse": float(max(mse_scores)),
    }
