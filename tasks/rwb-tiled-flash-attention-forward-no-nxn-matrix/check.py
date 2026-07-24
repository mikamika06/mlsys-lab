import numpy as np

def _full_attention_ref(Q, K, V):
    """Float64 full-softmax attention — the oracle."""
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)
    S_max = S.max(axis=1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)

    cases = [
        (128, 32, 32, 32),  # evenly divisible
        (100, 64, 16, 32),  # n not divisible by block_size
        (256, 16, 48, 64),  # larger n, non-square d
        (64, 32, 32, 64),   # block_size >= n  (single block)
        (200, 32, 32, 37),  # prime-ish block size
    ]

    max_err = 0.0
    for n, d_k, d_v, bs in cases:
        Q = rng.randn(n, d_k).astype(np.float64)
        K = rng.randn(n, d_k).astype(np.float64)
        V = rng.randn(n, d_v).astype(np.float64)

        ref = _full_attention_ref(Q, K, V)
        try:
            got = sol.tiled_flash_attention_forward(Q, K, V, block_size=bs)
        except Exception:
            return {"max_abs_err": float("inf")}

        got = np.asarray(got, dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
