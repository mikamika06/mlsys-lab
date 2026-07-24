import numpy as np

def _reference(mask):
    """Compute reference validity for a binary mask."""
    if mask.ndim == 0:
        raise ValueError("mask must have at least one dimension")
    if mask.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    groups = mask.reshape(*mask.shape[:-1], -1, 4)
    sums = np.sum(groups, axis=-1)
    group_validity = (sums == 2)
    overall = bool(np.all(group_validity))
    return group_validity, overall

def grade(sol, fx) -> dict:
    """Grade the candidate implementation."""
    # Generate a variety of test masks
    rng = np.random.default_rng(0)
    shapes = [
        (8, 12),          # 3 groups per row
        (3, 4, 8),        # 2 groups per last dim
        (5, 16),          # 4 groups per row
        (1, 4),           # single group
    ]
    ok = 1.0
    for shape in shapes:
        mask = rng.integers(0, 2, size=shape, dtype=np.uint8)
        try:
            got_group_validity, got_overall = sol.classify_mask_2_4(mask)
        except Exception:
            return {"exact_match": 0.0}
        ref_group_validity, ref_overall = _reference(mask)
        if not (isinstance(got_overall, bool) and
                np.array_equal(got_group_validity, ref_group_validity) and
                got_overall == ref_overall):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
