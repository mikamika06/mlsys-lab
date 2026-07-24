import numpy as np

def _ref(preds, warp_size):
    preds = np.asarray(preds)
    n = len(preds)
    if n % warp_size != 0:
        raise ValueError("Length not multiple of warp_size")
    reshaped = preds.reshape(-1, warp_size)
    out = np.empty(reshaped.shape[0], dtype=int)
    for i, block in enumerate(reshaped):
        out[i] = len(np.unique(block))
    return out

def grade(sol, fx) -> dict:
    cases = [
        (np.array([True]*32), 32),
        (np.array([False]*64), 32),
        (np.arange(64)%2, 32),
        (np.random.randint(0,5,size=96), 32),
        (np.array([1]*16 + [2]*8 + [3]*8), 32),
    ]
    ok = 1.0
    for preds, warp_size in cases:
        try:
            got = sol.warp_divergence_branch_count(preds, warp_size)
            ref = _ref(preds, warp_size)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray) or got.shape != ref.shape or not np.issubdtype(got.dtype, np.integer):
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
