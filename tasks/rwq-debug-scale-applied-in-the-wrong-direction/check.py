import numpy as np
from mlsys import scorers


def grade(sol, fx) -> dict:
    """
    Random (X, W, s) triples; verifies the X @ W invariant is preserved by
    the returned (X_prime, W_prime) migration-scale transform.
    """
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        b = int(rng.integers(2, 8))
        d_in = int(rng.integers(2, 10))
        d_out = int(rng.integers(2, 8))

        X = rng.standard_normal((b, d_in))
        W = rng.standard_normal((d_in, d_out))
        s = rng.uniform(0.2, 5.0, size=d_in)

        expected = X @ W
        try:
            got = sol.apply_migration_scale(X.copy(), W.copy(), s.copy())
            if not isinstance(got, tuple) or len(got) != 2:
                return {"max_abs_err": float("inf")}
            X_prime, W_prime = got
            X_prime = np.asarray(X_prime, dtype=np.float64)
            W_prime = np.asarray(W_prime, dtype=np.float64)
            if X_prime.shape != X.shape or W_prime.shape != W.shape:
                return {"max_abs_err": float("inf")}
            reconstructed = X_prime @ W_prime
            err = scorers.max_abs_err(expected, reconstructed)
        except Exception:
            return {"max_abs_err": float("inf")}

        worst = max(worst, err)

    return {"max_abs_err": worst}
