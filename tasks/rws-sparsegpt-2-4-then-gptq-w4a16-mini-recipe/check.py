import numpy as np


def _sparsegpt_2_4(W, X, lam_prune):
    """Structured 2:4 pruning with Hessian saliency + inverse-Hessian
    compensation of the surviving weights in each row group of 4."""
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = 2.0 * X @ X.T + lam_prune * np.eye(n)
    Hinv = np.linalg.inv(H)

    for r in range(m):
        for start in range(0, n, 4):
            cols = list(range(start, start + 4))
            scores = [(W[r, c] ** 2) / Hinv[c, c] for c in cols]
            keep = set(cols)
            for c, _ in sorted(zip(cols, scores), key=lambda x: x[1])[:2]:
                keep.remove(c)
            pruned = [c for c in cols if c not in keep]
            for c in pruned:
                old = W[r, c]
                for k in cols:
                    if k in keep:
                        W[r, k] -= old * Hinv[k, c] / Hinv[c, c]
                W[r, c] = 0.0
    return W


def _gptq_quantize(W, X, bits, damp):
    """Hessian-ordered per-column int-`bits` quantization with a
    per-row symmetric scale and inverse-Hessian error compensation."""
    W = np.asarray(W, dtype=np.float64).copy()
    m, n = W.shape
    H = X @ X.T
    H = H + np.eye(n) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)

    maxq = (1 << (bits - 1)) - 1
    row_scale = np.max(np.abs(W), axis=1) / maxq
    row_scale = np.where(row_scale == 0.0, 1.0, row_scale)

    W_q = np.zeros_like(W)
    for i in range(n):
        q = np.clip(np.round(W[:, i] / row_scale), -maxq, maxq) * row_scale
        W_q[:, i] = q
        err = q - W[:, i]
        if i + 1 < n:
            coeff = Hinv[i, i + 1:] / Hinv[i, i]
            W[:, i + 1:] -= np.outer(err, coeff)
    return W_q


def _oracle(W, X, bits=4, lam_prune=1e-2, damp=1e-2):
    W_sparse = _sparsegpt_2_4(W, X, lam_prune)
    return _gptq_quantize(W_sparse, X, bits, damp)


def _case(rng, m, n, s):
    W = rng.standard_normal((m, n))
    A = np.eye(n) + 0.4 * rng.standard_normal((n, n))
    Z = rng.standard_normal((s, n))
    X = (Z @ A).T  # (n, s): rows=features, cols=samples
    return W, X


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        _case(rng, 6, 8, 12),
        _case(rng, 5, 8, 20),
        _case(rng, 4, 12, 9),
    ]

    worst_out = 0.0
    worst_w = 0.0
    for W, X in cases:
        ref = _oracle(W, X)
        ref_Y = ref @ X

        try:
            got = sol.sparsegpt_then_gptq(np.asarray(W).copy(), np.asarray(X).copy())
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf"), "w_rel_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf"), "w_rel_err": float("inf")}

        got_Y = got @ X
        out_err = float(np.linalg.norm(got_Y - ref_Y) / (np.linalg.norm(ref_Y) + 1e-12))
        w_err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
        worst_out = max(worst_out, out_err)
        worst_w = max(worst_w, w_err)

    return {"rel_err": worst_out, "w_rel_err": worst_w}
