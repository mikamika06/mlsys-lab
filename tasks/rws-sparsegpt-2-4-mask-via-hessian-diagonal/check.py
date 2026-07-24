import numpy as np

from mlsys import scorers


def _oracle(W, X, damp):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    O, I = W.shape

    H = X.T @ X
    H = H + damp * np.mean(np.diag(H)) * np.eye(I)
    Hinv = np.linalg.inv(H)
    diag_hinv = np.diag(Hinv)

    mask = np.ones_like(W, dtype=np.int64)
    for o in range(O):
        for g0 in range(0, I, 4):
            idx = np.arange(g0, g0 + 4)
            scores = W[o, idx] ** 2 / diag_hinv[idx]
            order = np.argsort(scores, kind="stable")
            prune = idx[order[:2]]
            mask[o, prune] = 0

    W_hat = W.copy()
    for o in range(O):
        S = np.where(mask[o] == 0)[0]
        w_S = W[o, S]
        Hinv_SS = Hinv[np.ix_(S, S)]
        delta = -(Hinv[:, S] @ np.linalg.solve(Hinv_SS, w_S))
        W_hat[o] = W[o] + delta
        W_hat[o, S] = 0.0

    return mask, W_hat


def _build_cases():
    cases = []
    for seed, O, I, n, damp in [(0, 5, 16, 40, 0.01), (1, 4, 24, 60, 0.02), (2, 6, 12, 25, 0.005)]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((O, I))
        X = rng.standard_normal((n, I))
        cases.append((W, X, damp))
    return cases


def grade(sol, fx) -> dict:
    mask_ok = 1.0
    valid_frac_num = 0
    valid_frac_den = 0
    worst_what_rel = 0.0

    for W, X, damp in _build_cases():
        mask_ref, W_hat_ref = _oracle(W, X, damp)

        try:
            out = sol.sparsegpt_24_prune(W.copy(), X.copy(), damp=damp)
            mask_got, W_hat_got = out
            mask_got = np.asarray(mask_got)
            W_hat_got = np.asarray(W_hat_got, dtype=np.float64)
        except Exception:
            return {"mask_exact_match": 0.0, "valid_24_fraction": 0.0, "what_rel_err": float("inf")}

        if mask_got.shape != mask_ref.shape or W_hat_got.shape != W_hat_ref.shape:
            return {"mask_exact_match": 0.0, "valid_24_fraction": 0.0, "what_rel_err": float("inf")}
        if not np.all(np.isfinite(W_hat_got)):
            return {"mask_exact_match": 0.0, "valid_24_fraction": 0.0, "what_rel_err": float("inf")}

        if not np.array_equal(mask_got.astype(np.int64), mask_ref):
            mask_ok = 0.0

        O, I = mask_got.shape
        for o in range(O):
            for g0 in range(0, I, 4):
                valid_frac_den += 1
                if int(np.sum(mask_got[o, g0:g0 + 4])) == 2:
                    valid_frac_num += 1

        worst_what_rel = max(worst_what_rel, scorers.rel_err(W_hat_ref, W_hat_got))

    return {
        "mask_exact_match": mask_ok,
        "valid_24_fraction": (valid_frac_num / valid_frac_den) if valid_frac_den else 0.0,
        "what_rel_err": worst_what_rel,
    }
