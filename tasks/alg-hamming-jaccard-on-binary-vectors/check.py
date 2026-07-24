import numpy as np
from mlsys.scorers import rel_err

def _reference(B):
    B = np.asarray(B, dtype=np.uint8)
    B_bool = B.astype(bool)
    H_ref = np.sum(B_bool[:, None] != B_bool[None, :], axis=2).astype(np.int64)
    inter = (B_bool[:, None] & B_bool[None, :]).sum(axis=2)
    union = (B_bool[:, None] | B_bool[None, :]).sum(axis=2)
    J_ref = np.empty_like(inter, dtype=np.float64)
    mask = union == 0
    if np.any(mask):
        J_ref[mask] = 1.0
    J_ref[~mask] = inter[~mask] / union[~mask].astype(np.float64)
    return H_ref, J_ref

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    n, d = 10, 15
    B = rng.integers(0, 2, size=(n,d), dtype=np.uint8)
    try:
        H, J = sol.hamming_and_jaccard(B)
    except Exception:
        return {"exact_match": 0.0}
    H_ref, J_ref = _reference(B)
    ok_h = np.array_equal(H, H_ref)
    ok_j = rel_err(J, J_ref) <= 1e-12
    ok = 1.0 if (ok_h and ok_j) else 0.0
    return {"exact_match": ok}
