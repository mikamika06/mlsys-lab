import numpy as np

SPARSITY = 0.5
LAM = 0.01


def _mse(W, Wh, X):
    Y = W @ X
    Yh = Wh @ X
    return float(np.mean((Y - Yh) ** 2))


def _magnitude(W, sparsity):
    out = np.asarray(W, dtype=np.float64).copy()
    remove = int(out.size * sparsity)
    idx = np.argsort(np.abs(out).ravel())[:remove]
    for k in idx:
        i, j = divmod(int(k), out.shape[1])
        out[i, j] = 0.0
    return out


def _wanda(W, X, sparsity):
    out = np.asarray(W, dtype=np.float64).copy()
    z = np.linalg.norm(X, axis=1)
    scores = np.abs(out) * z[None, :]
    remove = int(out.size * sparsity)
    idx = np.argsort(scores.ravel())[:remove]
    for k in idx:
        i, j = divmod(int(k), out.shape[1])
        out[i, j] = 0.0
    return out


def _sparsegpt(W, X, sparsity, lam):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, d = W_work.shape
    H = 2.0 * X @ X.T + lam * np.eye(d)
    L = np.linalg.cholesky(H)
    Hinv = np.linalg.inv(L.T) @ np.linalg.inv(L)

    remove = int(W_work.size * sparsity)
    scores = (W_work * W_work) / (np.diag(Hinv)[None, :])
    flat = np.argsort(scores.ravel())[:remove]

    live = np.ones_like(W_work, dtype=bool)
    for idx in flat:
        i, j = divmod(int(idx), d)
        if not live[i, j]:
            continue
        old = W_work[i, j]
        live[i, j] = False
        denom = Hinv[j, j]
        W_work[i, :] += -old * Hinv[j, :] / denom
        W_work[i, j] = 0.0
    return W_work


def _oracle(W, X, sparsity, lam):
    W_mag = _magnitude(W, sparsity)
    W_wanda = _wanda(W, X, sparsity)
    W_sgpt = _sparsegpt(W, X, sparsity, lam)
    return {
        "mse_magnitude": _mse(W, W_mag, X),
        "mse_wanda": _mse(W, W_wanda, X),
        "mse_sparsegpt": _mse(W, W_sgpt, X),
    }


def grade(sol, fx) -> dict:
    W = fx["layer_w"]
    X = fx["layer_x"]
    ref = _oracle(W, X, SPARSITY, LAM)

    try:
        got = sol.compare_prune_methods_mse(W.copy(), X.copy(), SPARSITY, LAM)
        got_mag = float(got["mse_magnitude"])
        got_wanda = float(got["mse_wanda"])
        got_sgpt = float(got["mse_sparsegpt"])
    except Exception:
        return {
            "magnitude_err": float("inf"),
            "wanda_err": float("inf"),
            "sparsegpt_err": float("inf"),
            "ordering_ok": 0.0,
        }

    def rel(got_v, ref_v):
        return abs(got_v - ref_v) / (abs(ref_v) + 1e-9)

    ordering_ok = 1.0 if (got_sgpt <= got_wanda <= got_mag) else 0.0

    return {
        "magnitude_err": rel(got_mag, ref["mse_magnitude"]),
        "wanda_err": rel(got_wanda, ref["mse_wanda"]),
        "sparsegpt_err": rel(got_sgpt, ref["mse_sparsegpt"]),
        "ordering_ok": ordering_ok,
    }
