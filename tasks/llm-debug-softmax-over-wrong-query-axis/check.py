import numpy as np
from mlsys.scorers import max_abs_err

def _ref_sdpa(query, key, value, scale=None):
    """Reference implementation using NumPy only."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("All inputs must be 3‑D arrays.")
    B, Nq, dk = query.shape
    _, Nk, _ = key.shape
    _, _, dv = value.shape
    if key.shape[2] != dk or value.shape[1] != Nk:
        raise ValueError("Incompatible shapes.")
    if scale is None:
        scale = 1.0 / np.sqrt(dk)
    scores = query @ key.transpose(0, 2, 1) * scale
    # softmax over last axis (keys)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs @ value

def grade(sol, fx) -> dict:
    """Grade the student's sdpa implementation."""
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        B = rng.integers(1, 4)
        Nq = rng.integers(2, 6)
        Nk = rng.integers(2, 6)
        dk = rng.integers(3, 8)
        dv = rng.integers(2, 5)

        Q = rng.standard_normal((B, Nq, dk))
        K = rng.standard_normal((B, Nk, dk))
        V = rng.standard_normal((B, Nk, dv))

        try:
            ref = _ref_sdpa(Q, K, V)
            got = sol.sdpa(Q.tolist(), K.tolist(), V.tolist())
            got_arr = np.array(got)
        except Exception as e:
            return {"max_abs_err": 1e6}

        if got_arr.shape != ref.shape:
            return {"max_abs_err": 1e6}
        err = max_abs_err(ref, got_arr)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
