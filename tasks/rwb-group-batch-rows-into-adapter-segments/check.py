import numpy as np

def _reference(adapter_ids):
    # Stable sort
    perm = np.argsort(adapter_ids, kind='stable')
    sorted_ids = adapter_ids[perm]
    unique_ids, counts = np.unique(sorted_ids, return_counts=True)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
    return perm.astype(np.int64), offsets

def grade(sol, fx) -> dict:
    cases = [
        np.array([], dtype=np.int64),
        np.array([5], dtype=np.int64),
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([2, 0, 1, 2, 1], dtype=np.int64),
        np.array([3, 3, 3, 3], dtype=np.int64),
        np.random.randint(0, 10, size=20, dtype=np.int64),
    ]
    ok = 1.0
    for ids in cases:
        try:
            got_perm, got_offsets = sol.group_rows_by_adapter(ids)
            ref_perm, ref_offsets = _reference(ids)
        except Exception:
            return {"exact_match": 0.0}
        if not (np.array_equal(got_perm, ref_perm) and np.array_equal(got_offsets, ref_offsets)):
            ok = 0.0
            break
    return {"exact_match": ok}
