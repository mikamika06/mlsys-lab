import numpy as np

def _reference(gradients, mask, grow_count):
    gradients = np.asarray(gradients)
    mask = np.asarray(mask, dtype=bool)
    assert gradients.shape == mask.shape
    zero_pos = np.where(~mask)[0]
    if len(zero_pos) == 0:
        return mask.copy()
    abs_grad = np.abs(gradients[zero_pos])
    if grow_count > len(zero_pos):
        grow_count = len(zero_pos)
    # argpartition gives unsorted indices; we only need the top k
    top_idx_within_zero = np.argpartition(-abs_grad, grow_count-1)[:grow_count]
    top_indices = zero_pos[top_idx_within_zero]
    new_mask = mask.copy()
    new_mask[top_indices] = True
    return new_mask

def grade(sol, fx) -> dict:
    # generate random test cases
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(20):
        n = rng.integers(5, 50)
        gradients = rng.standard_normal(n).astype(np.float64)
        mask = rng.choice([False, True], size=n, p=[0.7, 0.3])
        grow_count = rng.integers(1, max(2, int(mask.sum()*0.5)+1))
        try:
            got = sol.rigl_grow_step(gradients, mask, grow_count)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray):
            return {"exact_match": 0.0}
        ref = _reference(gradients, mask, grow_count)
        if got.shape != ref.shape or got.dtype != ref.dtype:
            return {"exact_match": 0.0}
        if not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
