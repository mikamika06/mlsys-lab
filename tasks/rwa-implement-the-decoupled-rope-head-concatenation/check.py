import numpy as np
from mlsys import scorers

def _ref(q_lat, k_lat, q_rope, k_rope):
    """NumPy oracle: concatenation → scaled dot-product → softmax."""
    B, H, N, D_l = q_lat.shape
    D_r = q_rope.shape[-1]
    D = D_l + D_r
    Q = np.concatenate([q_lat, q_rope], axis=-1)   # (B, H, N, D)
    K = np.concatenate([k_lat, k_rope], axis=-1)   # (B, H, N, D)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D)  # (B, H, N, N)
    # softmax over last axis (keys)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    np.random.seed(42)
    cases = [
        (2, 3, 5, 64, 32),
        (4, 6, 10, 128, 64),
        (1, 2, 3, 16, 8),
    ]
    worst = 0.0
    for B, H, N, D_l, D_r in cases:
        ql = np.random.randn(B, H, N, D_l).astype(np.float64)
        kl = np.random.randn(B, H, N, D_l).astype(np.float64)
        qr = np.random.randn(B, H, N, D_r).astype(np.float64)
        kr = np.random.randn(B, H, N, D_r).astype(np.float64)
        try:
            got = sol.decoupled_rope_score(ql, kl, qr, kr)
        except Exception:
            return {"max_abs_err": 1e6}
        ref = _ref(ql, kl, qr, kr)
        err = scorers.max_abs_err(ref, got)
        worst = max(worst, err)
    return {"max_abs_err": worst}
