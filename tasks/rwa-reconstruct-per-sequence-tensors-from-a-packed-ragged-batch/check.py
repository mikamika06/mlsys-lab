import numpy as np

def _reference(packed, cu_seqlens):
    return [packed[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]

def grade(sol, fx) -> dict:
    # If no fixtures provided, generate deterministic random data
    if "packed" not in fx or "cu_seqlens" not in fx:
        rng = np.random.default_rng(0)
        S = 4
        D = 3
        lengths = rng.integers(1, 5, size=S)
        N = lengths.sum()
        packed = rng.standard_normal((N, D))
        cu_seqlens = np.concatenate([[0], np.cumsum(lengths)])
        fx = {"packed": packed, "cu_seqlens": cu_seqlens}
    try:
        got = sol.unpack_sequences(fx["packed"], fx["cu_seqlens"])
    except Exception:
        return {"exact_match": 0.0}
    ref = _reference(fx["packed"], fx["cu_seqlens"])
    if len(got) != len(ref):
        return {"exact_match": 0.0}
    for a, b in zip(ref, got):
        if not np.array_equal(a, b):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
