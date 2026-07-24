import numpy as np


def _oracle_run_with_cache(Wq, Wk, Wv, X, kv_cache):
    d = X.shape[1]
    if kv_cache is None:
        K_prefix = np.zeros((0, d), dtype=np.float64)
        V_prefix = np.zeros((0, d), dtype=np.float64)
    else:
        K_prefix = np.asarray(kv_cache["K"], dtype=np.float64)
        V_prefix = np.asarray(kv_cache["V"], dtype=np.float64)
    P = K_prefix.shape[0]

    Q_new = X @ Wq
    K_new = X @ Wk
    V_new = X @ Wv
    K_all = np.concatenate([K_prefix, K_new], axis=0)
    V_all = np.concatenate([V_prefix, V_new], axis=0)

    L = X.shape[0]
    out = np.zeros((L, d), dtype=np.float64)
    for i in range(L):
        end = P + i + 1
        scores = (Q_new[i] @ K_all[:end].T) / np.sqrt(d)
        scores = scores - np.max(scores)
        w = np.exp(scores)
        w = w / np.sum(w)
        out[i] = w @ V_all[:end]

    return out, {"K": K_all, "V": V_all}


def _oracle_dense(Wq, Wk, Wv, X_full):
    S, d = X_full.shape
    Q = X_full @ Wq
    K = X_full @ Wk
    V = X_full @ Wv
    scores = (Q @ K.T) / np.sqrt(d)
    qi = np.arange(S)[:, None]
    kj = np.arange(S)[None, :]
    scores = np.where(kj > qi, -np.inf, scores)
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    configs = [(4, 3, 2), (6, 5, 4), (8, 6, 3), (5, 2, 5)]

    worst_err = 0.0
    bookkeeping_ok = 1.0

    for d, P, L in configs:
        Wq = rng.standard_normal((d, d)) * 0.5
        Wk = rng.standard_normal((d, d)) * 0.5
        Wv = rng.standard_normal((d, d)) * 0.5
        X1 = rng.standard_normal((P, d))
        X2 = rng.standard_normal((L, d))
        X_full = np.concatenate([X1, X2], axis=0)

        dense = _oracle_dense(Wq, Wk, Wv, X_full)
        ref_out1, ref_cache1 = _oracle_run_with_cache(Wq, Wk, Wv, X1, None)
        ref_out2, ref_cache2 = _oracle_run_with_cache(Wq, Wk, Wv, X2, ref_cache1)

        try:
            got_out1, got_cache1 = sol.run_with_cache(Wq, Wk, Wv, X1.copy(), None)
            got_out2, got_cache2 = sol.run_with_cache(Wq, Wk, Wv, X2.copy(), got_cache1)
        except Exception:
            return {"max_abs_err": float("inf"), "bookkeeping_exact": 0.0}

        got_out1 = np.asarray(got_out1, dtype=np.float64)
        got_out2 = np.asarray(got_out2, dtype=np.float64)

        if got_out1.shape != ref_out1.shape or got_out2.shape != ref_out2.shape:
            return {"max_abs_err": float("inf"), "bookkeeping_exact": 0.0}
        if not (np.all(np.isfinite(got_out1)) and np.all(np.isfinite(got_out2))):
            return {"max_abs_err": float("inf"), "bookkeeping_exact": 0.0}

        err1 = float(np.max(np.abs(got_out1 - dense[:P])))
        err2 = float(np.max(np.abs(got_out2 - dense[P:])))
        worst_err = max(worst_err, err1, err2)

        try:
            got_K2 = np.asarray(got_cache2["K"], dtype=np.float64)
            got_V2 = np.asarray(got_cache2["V"], dtype=np.float64)
        except Exception:
            return {"max_abs_err": worst_err, "bookkeeping_exact": 0.0}

        if got_K2.shape != (P + L, d) or got_V2.shape != (P + L, d):
            bookkeeping_ok = 0.0
        else:
            if not np.allclose(got_K2, ref_cache2["K"], atol=1e-8):
                bookkeeping_ok = 0.0
            if not np.allclose(got_V2, ref_cache2["V"], atol=1e-8):
                bookkeeping_ok = 0.0

    return {"max_abs_err": worst_err, "bookkeeping_exact": bookkeeping_ok}
