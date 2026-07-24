import numpy as np

def _reference(shape, strides, newshape):
    # Compute minimal buffer size in elements
    itemsize = np.int64().itemsize
    max_offset = sum((s - 1) * st for s, st in zip(shape, strides))
    total_bytes = max_offset + itemsize
    n_elements = (total_bytes + itemsize - 1) // itemsize
    base = np.arange(n_elements, dtype=np.int64)
    try:
        arr = np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides)
        reshaped = arr.reshape(newshape)
        return np.may_share_memory(arr, reshaped)
    except Exception:
        return False

def grade(sol, fx) -> dict:
    cases = [
        ((4,), (8,),   (2, 2)),   # contiguous → view
        ((4,), (16,),  (2, 2)),   # non‑contiguous → copy
        ((3, 3), (24, 8), (5, 5)),# incompatible reshape
        ((6,), (8,),   (2, 3)),   # contiguous → view
        ((4, 2), (16, 8), (8,)), # contiguous → view
        ((4, 2), (24, 8), (8,)), # non‑contiguous → copy
    ]
    ok = 1.0
    for shape, strides, newshape in cases:
        try:
            got = sol.can_reshape_view(shape, strides, newshape)
            ref = _reference(shape, strides, newshape)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
