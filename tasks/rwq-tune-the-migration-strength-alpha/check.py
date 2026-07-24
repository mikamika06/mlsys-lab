import numpy as np

EPS = 1e-8
ALPHAS = np.linspace(0.0, 1.0, 11)


def _w8a8_mse(W, X, alpha):
    amax_x = np.max(np.abs(X), axis=0)
    amax_w = np.max(np.abs(W), axis=0)
    s = np.power(amax_x, alpha) / (np.power(amax_w, 1.0 - alpha) + EPS)
    s = np.maximum(s, EPS)

    Xs = X / s[None, :]
    Ws = W * s[None, :]

    sx = np.max(np.abs(Xs))
    sx = sx / 127.0 if sx > 0 else 1.0
    Xq = np.clip(np.round(Xs / sx), -127, 127) * sx

    aw = np.max(np.abs(Ws), axis=1)
    sw = np.where(aw > 0, aw / 127.0, 1.0)
    Wq = np.clip(np.round(Ws / sw[:, None]), -127, 127) * sw[:, None]

    Yhat = Xq @ Wq.T
    Y = X @ W.T
    return float(np.mean((Yhat - Y) ** 2))


def _oracle_sweep(W, X, alphas):
    mses = [_w8a8_mse(W, X, a) for a in alphas]
    idx = int(np.argmin(mses))
    return idx, mses[idx]


def _make_trial(rng, d_out, d_in, n_cal, n_outlier, mag):
    X = rng.normal(size=(n_cal, d_in))
    outlier_ch = rng.choice(d_in, size=n_outlier, replace=False)
    X[:, outlier_ch] *= mag
    W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 1.5, size=(1, d_in))
    return W, X


def grade(sol, fx) -> dict:
    """
    Builds seeded random (W, X) trials with a couple of outlier
    activation channels (the classic SmoothQuant scenario), sweeps a
    shared alpha grid recomputing the per-channel smoothing scale, the
    W8A8 (dynamic per-tensor activation, per-channel weight) int8
    quantization, and the resulting output MSE, with a NumPy oracle.
    Compares the submission's argmin alpha index (exact) and MSE at that
    index (relative error) to the oracle's.
    """
    rng = np.random.default_rng(0)
    idx_ok = 1.0
    mse_rel_worst = 0.0
    for _ in range(3):
        W, X = _make_trial(rng, d_out=5, d_in=12, n_cal=40, n_outlier=2, mag=20.0)
        idx_exp, mse_exp = _oracle_sweep(W, X, ALPHAS)

        try:
            idx_got, mse_got = sol.sweep_alpha(W.copy(), X.copy(), ALPHAS.copy())
            idx_got = int(idx_got)
            mse_got = float(mse_got)
        except Exception:
            return {"argmin_index_match": 0.0, "mse_rel_err": float("inf")}

        if idx_got != idx_exp:
            idx_ok = 0.0
        mse_rel = abs(mse_got - mse_exp) / (abs(mse_exp) + 1e-12)
        mse_rel_worst = max(mse_rel_worst, mse_rel)

    return {"argmin_index_match": idx_ok, "mse_rel_err": mse_rel_worst}
