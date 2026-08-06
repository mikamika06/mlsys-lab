import numpy as np

def _softmax(x):
    """Numerically stable softmax over the last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def _reference(Q, K, V):
    # Standard scaled dot‑product attention
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])
    weights = _softmax(scores)
    return weights @ V

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.standard_normal((2, 4, 8)), rng.standard_normal((2, 4, 8)), rng.standard_normal((2, 4, 6))),
        (rng.standard_normal((3, 5, 16)), rng.standard_normal((3, 5, 16)), rng.standard_normal((3, 5, 12))),
        (rng.standard_normal((1, 7, 10)), rng.standard_normal((1, 7, 10)), rng.standard_normal((1, 7, 8))),
    ]
    max_err = 0.0
    for Q_arr, K_arr, V_arr in cases:
        try:
            cand = sol.gqa_limit_nkv_1(Q_arr.tolist(), K_arr.tolist(), V_arr.tolist())
            cand_arr = np.array(cand)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        ref = _reference(Q_arr, K_arr, V_arr)
        err = np.max(np.abs(cand_arr - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
