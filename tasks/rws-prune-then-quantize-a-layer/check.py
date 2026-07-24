import numpy as np
from mlsys import scorers

def _oracle(W, group_size):
    """Reference implementation of prune‑then‑quantize."""
    W = np.asarray(W, dtype=np.float64)
    n, d = W.shape
    out = np.zeros_like(W)
    for i in range(n):
        row = W[i]
        for start in range(0, d, group_size):
            end = min(start + group_size, d)
            block = row[start:end].copy()
            if len(block) <= 2:
                keep_idx = np.arange(len(block))
            else:
                # indices of two largest absolute values
                keep_idx = np.argpartition(np.abs(block), -2)[-2:]
            mask = np.zeros_like(block, dtype=bool)
            mask[keep_idx] = True
            block[~mask] = 0.0
            max_abs = np.max(np.abs(block))
            if max_abs == 0:
                out[i, start:end] = 0.0
                continue
            scale = max_abs / 7.0
            q = np.round(block / scale).clip(-8, 7)
            out[i, start:end] = q * scale
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.standard_normal((8, 12)), 4),
        (rng.standard_normal((5, 10)), 4),
        (rng.standard_normal((7, 9)), 3),   # non‑divisible group size
        (np.zeros((4, 8)), 4),              # all zeros
        (rng.standard_normal((6, 14)), 4)
    ]
    max_err = 0.0
    for W, g in cases:
        try:
            cand = sol.prune_then_quantize(W, g)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        oracle = _oracle(W, g)
        err = scorers.max_abs_err(oracle, cand)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
