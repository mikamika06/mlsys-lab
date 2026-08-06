import numpy as np

def _reference_classify(arr):
    c = arr.flags.c_contiguous
    f = arr.flags.f_contiguous
    if c and not f:
        return "C"
    elif f and not c:
        return "F"
    else:
        return "Neither"

def grade(sol, fx) -> dict:
    # Test cases covering all three categories
    tests = []

    # C contiguous: normal reshape
    a = np.arange(12).reshape((3, 4))
    tests.append(a)

    # F contiguous: transpose
    b = a.T
    tests.append(b)

    # Neither: slice with step >1
    c = a[::2]
    tests.append(c)

    # Neither: complex stride via advanced indexing
    d = np.arange(24).reshape((4, 6))[::2, ::3]
    tests.append(d)

    # Neither: 1‑D array (both flags true)
    e = np.arange(5)
    tests.append(e.reshape((5, 1)))

    ok = 1.0
    for arr in tests:
        try:
            got = sol.classify_contiguity(arr.tolist())
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_classify(arr)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
