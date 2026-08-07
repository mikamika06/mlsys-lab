import math
import random
import numpy as np
from mlsys import scorers

def _ref(q_lat, k_lat, q_rope, k_rope):
    ql = np.array(q_lat, dtype=np.float64)
    kl = np.array(k_lat, dtype=np.float64)
    qr = np.array(q_rope, dtype=np.float64)
    kr = np.array(k_rope, dtype=np.float64)
    B, H, N, D_l = ql.shape
    D_r = qr.shape[-1]
    D = D_l + D_r
    Q = np.concatenate([ql, qr], axis=-1)   # (B, H, N, D)
    K = np.concatenate([kl, kr], axis=-1)   # (B, H, N, D)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D)  # (B, H, N, N)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    random.seed(42)
    cases = [
        (2, 3, 5, 16, 8),
        (2, 2, 4, 8, 4),
    ]
    worst = 0.0
    for B, H, N, D_l, D_r in cases:
        ql = [[[[random.gauss(0, 1) for _ in range(D_l)] for _ in range(N)] for _ in range(H)] for _ in range(B)]
        kl = [[[[random.gauss(0, 1) for _ in range(D_l)] for _ in range(N)] for _ in range(H)] for _ in range(B)]
        qr = [[[[random.gauss(0, 1) for _ in range(D_r)] for _ in range(N)] for _ in range(H)] for _ in range(B)]
        kr = [[[[random.gauss(0, 1) for _ in range(D_r)] for _ in range(N)] for _ in range(H)] for _ in range(B)]
        try:
            got = sol.decoupled_rope_score(ql, kl, qr, kr)
        except Exception:
            return {"max_abs_err": 1e6}
        ref = _ref(ql, kl, qr, kr)
        err = scorers.max_abs_err(ref, got)
        worst = max(worst, err)
    return {"max_abs_err": worst}
