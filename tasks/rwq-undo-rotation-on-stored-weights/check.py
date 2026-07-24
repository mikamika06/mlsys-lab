import numpy as np

def grade(sol, fx):
    rng = np.random.RandomState(42)
    max_err = 0.0
    for _ in range(5):
        m = rng.randint(2, 8)
        n = rng.randint(2, 8)
        # orthogonal H
        A = rng.randn(m, m)
        H, _ = np.linalg.qr(A)
        # small original weights
        W = (rng.rand(m, n) - 0.5) * 0.5   # range -0.25 .. 0.25
        W_rot = H @ W
        scale = 0.1
        zero_point = rng.randint(-5, 6)   # integer, provided as float
        W_quantized = np.round(W_rot / scale + zero_point).astype(np.int8)
        try:
            W_rec = sol.reconstruct_weights(W_quantized, H, scale, float(zero_point))
        except Exception:
            return {"max_abs_err": float("inf")}
        # reference computation
        W_dequant = (W_quantized.astype(np.float64) - zero_point) * scale
        W_oracle = H.T @ W_dequant
        err = float(np.max(np.abs(W_rec - W_oracle)))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
