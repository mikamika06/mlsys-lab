import numpy as np

def _reference(budgets, decode_counts):
    budgets = np.asarray(budgets)
    decode_counts = np.asarray(decode_counts)
    if budgets.shape != decode_counts.shape:
        raise ValueError("Shape mismatch")
    prefill = budgets - decode_counts
    prefill = np.maximum(prefill, 0)
    is_prefill = prefill > 0
    return prefill.astype(np.int64), is_prefill

def grade(sol, fx) -> dict:
    try:
        # The solution should expose a function named classify_prefill_decode
        got_prefill, got_is_prefill = sol.classify_prefill_decode(
            np.array([10, 5, 8]), np.array([3, 7, 2])
        )
        ref_prefill, ref_is_prefill = _reference(np.array([10, 5, 8]),
                                                 np.array([3, 7, 2]))
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0
    if not (np.array_equal(got_prefill, ref_prefill) and
            np.array_equal(got_is_prefill, ref_is_prefill)):
        ok = 0.0

    # Additional checks for shape mismatch or wrong dtype
    try:
        sol.classify_prefill_decode(np.array([1, 2]), np.array([3]))
    except ValueError:
        pass
    else:
        ok = 0.0

    return {"exact_match": ok}
