import numpy as np


def _masks(W, X, sparsity):
    d_out, d_in = W.shape
    n_prune = int(round(sparsity * d_in))
    col_norm = np.linalg.norm(X, axis=1)  # (d_in,), per input feature

    M_mag = np.ones_like(W)
    M_wanda = np.ones_like(W)
    for i in range(d_out):
        order_mag = np.argsort(np.abs(W[i]), kind="stable")
        M_mag[i, order_mag[:n_prune]] = 0.0

        metric = np.abs(W[i]) * col_norm
        order_w = np.argsort(metric, kind="stable")
        M_wanda[i, order_w[:n_prune]] = 0.0
    return M_wanda, M_mag


def _sq_frobenius_err(W, M, X):
    diff = W @ X - (W * M) @ X
    return float(np.sum(diff ** 2))


def _oracle(W, X, sparsity):
    M_wanda, M_mag = _masks(W, X, sparsity)
    e_wanda = _sq_frobenius_err(W, M_wanda, X)
    e_mag = _sq_frobenius_err(W, M_mag, X)
    return e_wanda, e_mag


def _make_trial(rng, d_out, d_in, n, n_outlier, mag):
    X = rng.normal(size=(d_in, n))
    outlier_ch = rng.choice(d_in, size=n_outlier, replace=False)
    X[outlier_ch, :] *= mag
    W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.2, 1.0, size=(1, d_in))
    return W, X


def grade(sol, fx) -> dict:
    """
    Builds seeded random (W, X) trials with a couple of outlier
    activation channels (rows of X), computes the Wanda mask
    (|W_ij| * ||X row j||_2, per output row) and the pure-magnitude mask
    (|W_ij| alone), both at the same sparsity, and the resulting squared
    Frobenius output-reconstruction error ||WX - (W*M)X||_F^2 for each,
    with a NumPy oracle. Compares the submission's two error values
    (relative error) to the oracle's, and checks Wanda never loses more
    output energy than pure magnitude pruning.
    """
    rng = np.random.default_rng(0)
    wanda_rel_worst = 0.0
    mag_rel_worst = 0.0
    ordering_ok = 1.0
    for _ in range(4):
        d_out = int(rng.integers(4, 8))
        d_in = int(rng.integers(12, 20))
        n = int(rng.integers(18, 32))
        W, X = _make_trial(rng, d_out, d_in, n, n_outlier=2, mag=15.0)
        sparsity = 0.5

        e_wanda_exp, e_mag_exp = _oracle(W, X, sparsity)

        try:
            e_wanda_got, e_mag_got = sol.wanda_vs_magnitude_error(W.copy(), X.copy(), sparsity)
            e_wanda_got = float(e_wanda_got)
            e_mag_got = float(e_mag_got)
        except Exception:
            return {"wanda_rel_err": float("inf"), "magnitude_rel_err": float("inf"), "wanda_le_magnitude": 0.0}

        wanda_rel_worst = max(wanda_rel_worst, abs(e_wanda_got - e_wanda_exp) / (abs(e_wanda_exp) + 1e-12))
        mag_rel_worst = max(mag_rel_worst, abs(e_mag_got - e_mag_exp) / (abs(e_mag_exp) + 1e-12))

        if e_wanda_got > e_mag_got + 1e-6:
            ordering_ok = 0.0

    return {
        "wanda_rel_err": wanda_rel_worst,
        "magnitude_rel_err": mag_rel_worst,
        "wanda_le_magnitude": ordering_ok,
    }
