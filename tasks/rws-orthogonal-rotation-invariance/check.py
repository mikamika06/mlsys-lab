import numpy as np


def _make_orthogonal(rng, n):
    A = rng.normal(size=(n, n))
    S = A + A.T
    _, Q = np.linalg.eigh(S)  # eigenvectors of a real symmetric matrix are orthonormal
    return Q


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(6):
        d_in = int(rng.integers(3, 9))
        d_out = int(rng.integers(2, 7))
        W = rng.normal(size=(d_out, d_in))
        x = rng.normal(size=d_in)
        Q = _make_orthogonal(rng, d_in)

        W_rot_exp = W @ Q
        x_rot_exp = Q.T @ x
        y_exp = W @ x  # rotation invariance: should equal W_rot_exp @ x_rot_exp

        try:
            W_rot_got, x_rot_got, y_got = sol.rotate_and_matvec(W.copy(), x.copy(), Q.copy())
            W_rot_got = np.asarray(W_rot_got, dtype=np.float64)
            x_rot_got = np.asarray(x_rot_got, dtype=np.float64)
            y_got = np.asarray(y_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if W_rot_got.shape != W_rot_exp.shape or x_rot_got.shape != x_rot_exp.shape or y_got.shape != y_exp.shape:
            return {"max_abs_err": float("inf")}

        worst = max(
            worst,
            float(np.max(np.abs(W_rot_got - W_rot_exp))),
            float(np.max(np.abs(x_rot_got - x_rot_exp))),
            float(np.max(np.abs(y_got - y_exp))),
        )

    return {"max_abs_err": worst}
