import random

def _reference(N, heads, d):
    s = 8  # bytes per float64
    naive = heads * N * N * s
    flash = 3 * heads * N * d * s
    return (naive, flash)

def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 1),
        (10, 2, 5),
        (32, 4, 8),
        (64, 8, 16),
        (128, 12, 24)
    ]
    ok = 1.0
    for N, heads, d in cases:
        try:
            got = sol.peak_attention_memory(N, heads, d)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, tuple) or len(got) != 2:
            return {"exact_match": 0.0}
        ref = _reference(N, heads, d)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
