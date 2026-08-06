import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _ref_mha(X, Wq, Wk, Wv, Wo, num_heads):
    B, S, E = X.shape
    d = E // num_heads

    q = X @ Wq
    k = X @ Wk
    v = X @ Wv

    q = q.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)
    k = k.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, num_heads, d).transpose(0, 2, 1, 3)

    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    weights = _softmax(scores)
    out = weights @ v

    out = out.transpose(0, 2, 1, 3).reshape(B, S, E)
    return out @ Wo


def grade(sol, fx) -> dict:
    cases = [
        (2, 5, 8, 2, 1),
        (1, 6, 12, 3, 2),
        (3, 4, 16, 4, 3),
    ]

    worst = 0.0
    for B, S, E, H, seed in cases:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(B, S, E)).astype(np.float64)
        Wq = rng.normal(size=(E, E)).astype(np.float64)
        Wk = rng.normal(size=(E, E)).astype(np.float64)
        Wv = rng.normal(size=(E, E)).astype(np.float64)
        Wo = rng.normal(size=(E, E)).astype(np.float64)

        ref = _ref_mha(X, Wq, Wk, Wv, Wo, H)

        X_list = X.tolist()
        Wq_list = Wq.tolist()
        Wk_list = Wk.tolist()
        Wv_list = Wv.tolist()
        Wo_list = Wo.tolist()

        try:
            got = np.asarray(sol.mha_forward(X_list, Wq_list, Wk_list, Wv_list, Wo_list, H))
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
