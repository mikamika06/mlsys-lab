import numpy as np


def _obs_oracle(W, X, sparsity, lam):
    W_work = np.asarray(W, dtype=np.float64).copy()
    m, d = W_work.shape
    H = 2.0 * X @ X.T + lam * np.eye(d)
    L = np.linalg.cholesky(H)
    Hinv = np.linalg.inv(L.T) @ np.linalg.inv(L)

    total = m * d
    remove = int(total * sparsity)
    scores = (W_work * W_work) / (np.diag(Hinv)[None, :])
    flat = np.argsort(scores.ravel())[:remove]

    mask = np.ones_like(W_work, dtype=bool)
    for idx in flat:
        i, j = divmod(int(idx), d)
        if not mask[i, j]:
            continue
        old = W_work[i, j]
        mask[i, j] = False
        W_work[i, j] = 0.0
        denom = Hinv[j, j]
        W_work[i, :] += -old * Hinv[j, :] / denom
        W_work[i, j] = 0.0

    return W_work, mask


def _magnitude_baseline(W, X, sparsity):
    out = np.asarray(W, dtype=np.float64).copy()
    remove = int(out.size * sparsity)
    idx = np.argsort(np.abs(out).ravel())[:remove]
    mask = np.ones_like(out, dtype=bool)
    for k in idx:
        i, j = divmod(int(k), out.shape[1])
        out[i, j] = 0.0
        mask[i, j] = False
    return out, mask


def grade(sol, fx) -> dict:
    W = np.array(
        [
            [0.8, -0.15, 1.2, 0.05],
            [-0.6, 0.4, -0.1, 0.9],
            [1.1, 0.07, -0.5, 0.3],
        ],
        dtype=np.float64,
    )
    X = np.array(
        [
            [1.0, 0.1, -0.2, 0.4],
            [0.3, -0.7, 0.5, 0.2],
            [-0.4, 0.6, 0.8, -0.1],
            [0.2, -0.3, 0.4, 0.9],
        ],
        dtype=np.float64,
    )
    sparsity = 0.5
    lam = 0.01

    ref, _ = _obs_oracle(W, X, sparsity, lam)
    base, _ = _magnitude_baseline(W, X, sparsity)

    try:
        got, mask = sol.sparsegpt_layerwise(W, X, sparsity, lam)
        got = np.asarray(got, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
    except Exception:
        return {"oracle_rel_err": float("inf"), "beats_magnitude": 0.0}

    oracle_err = np.linalg.norm((got - ref) @ X) / (np.linalg.norm(ref @ X) + 1e-12)
    got_err = np.linalg.norm((got - W) @ X)
    base_err = np.linalg.norm((base - W) @ X)

    return {
        "oracle_rel_err": float(oracle_err),
        "beats_magnitude": float(got_err < base_err),
    }
