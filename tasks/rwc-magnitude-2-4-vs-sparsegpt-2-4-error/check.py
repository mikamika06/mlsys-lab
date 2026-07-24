import numpy as np

from mlsys import scorers

LAMBDA = 1e-4


def _magnitude_prune_2_4(W: np.ndarray) -> np.ndarray:
    """Naive magnitude 2:4: per row, per group of 4 consecutive columns,
    zero the 2 smallest-magnitude entries, keep the 2 largest. No
    compensation of the kept weights.
    """
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    out = W.copy()
    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            block = W[r, cols]
            order = np.argsort(np.abs(block))  # ascending magnitude
            for c in [cols[order[0]], cols[order[1]]]:
                out[r, c] = 0.0
    return out


def _sparsegpt_2_4(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Hessian-aware SparseGPT 2:4: damped Hessian H = X^T X / s + lambda*I,
    per-row-group saliency S = w^2 / diag(H^-1), prune the 2 lowest-saliency
    weights per group of 4, and compensate the 2 kept weights via the
    inverse-Hessian update w_k -= w_j * Hinv[k,j] / Hinv[j,j].
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = X.shape[0]
    H = (X.T @ X) / s + LAMBDA * np.eye(X.shape[1])
    Hinv = np.linalg.inv(H)

    mask = np.zeros_like(W, dtype=np.int64)
    out = W.copy()

    for r in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            cols = list(range(start, start + 4))
            scores = [(W[r, c] ** 2) / Hinv[c, c] for c in cols]
            keep = set(cols)
            for c, _ in sorted(zip(cols, scores), key=lambda x: x[1])[:2]:
                keep.remove(c)
            for c in keep:
                mask[r, c] = 1

            pruned = [c for c in cols if mask[r, c] == 0]
            for c in pruned:
                old = out[r, c]
                for k in cols:
                    if mask[r, k] == 1:
                        out[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                out[r, c] = 0.0

    return out


def _rel_err(Y_true, Y_approx) -> float:
    return float(np.linalg.norm(Y_approx - Y_true) / np.linalg.norm(Y_true))


def _oracle(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y_true = X @ W.T

    W_mag = _magnitude_prune_2_4(W)
    err_mag = _rel_err(Y_true, X @ W_mag.T)

    W_sp = _sparsegpt_2_4(W, X)
    err_sp = _rel_err(Y_true, X @ W_sp.T)

    reduction = 1.0 - err_sp / err_mag
    return err_mag, err_sp, reduction


def _synthetic_case(rng, out_dim, in_dim, s, salient_scale):
    W = rng.normal(0.0, 1.0, size=(out_dim, in_dim))
    A = rng.normal(0.0, 1.0, size=(in_dim, in_dim)) * 0.3 + np.eye(in_dim)
    Z = rng.normal(0.0, 1.0, size=(s, in_dim))
    X = Z @ A
    n_salient = max(1, in_dim // 6)
    salient = rng.choice(in_dim, size=n_salient, replace=False)
    X[:, salient] *= salient_scale
    return W, X


def _synthetic_cases():
    rng = np.random.default_rng(19)
    return [
        _synthetic_case(rng, 6, 12, 48, 12.0),
        _synthetic_case(rng, 10, 20, 80, 20.0),
        _synthetic_case(rng, 4, 8, 32, 10.0),
    ]


def grade(sol, fx) -> dict:
    cases = [(fx["W"], fx["X"])] + _synthetic_cases()

    worst = 0.0
    for W, X in cases:
        ref = _oracle(W, X)
        try:
            got = sol.compare_magnitude_vs_sparsegpt_2_4(np.asarray(W).copy(), np.asarray(X).copy())
            got = tuple(float(v) for v in got)
        except Exception:
            return {"rel_err": float("inf")}

        if len(got) != 3 or not all(np.isfinite(v) for v in got):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(np.array(ref), np.array(got))
        worst = max(worst, err)

    return {"rel_err": worst}
