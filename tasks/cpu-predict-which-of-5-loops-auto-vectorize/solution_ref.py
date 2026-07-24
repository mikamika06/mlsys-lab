def classify_loops() -> list:
    """
    Return True/False for each of 5 loops: True = auto-vectorizable.

    Loop 0: a[i] = b[i] + c[i]          -> True  (no dep, uniform stride)
    Loop 1: a[i] = a[i-1] + b[i]        -> False (loop-carried dep)
    Loop 2: s += a[i]                    -> True  (simple reduction)
    Loop 3: a[i] = b[i] if b[i]>0 else 0 -> True  (branch-free select)
    Loop 4: a[i] = b[i*i % N]           -> False (non-uniform stride)
    """
    return [True, False, True, True, False]
