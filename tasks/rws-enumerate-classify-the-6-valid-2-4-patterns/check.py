import numpy as np

def _build_reference():
    # Generate all 4‑bit patterns and keep those with exactly two ones.
    patterns = [tuple(int(b) for b in format(i, '04b')) for i in range(16)]
    valid = sorted([p for p in patterns if sum(p) == 2])
    return {p: idx for idx, p in enumerate(valid)}

_REFERENCE_MAP = _build_reference()

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    # Generate a handful of test cases with random rows.
    for n_rows in range(5, 11):
        vecs = rng.integers(0, 2, size=(n_rows, 4))
        try:
            got = sol.classify_patterns(vecs)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, np.ndarray) or got.shape != (n_rows,):
            return {"exact_match": 0.0}
        # Build expected array.
        exp = np.full(n_rows, -1, dtype=int)
        mask = vecs.sum(axis=1) == 2
        for i, row in enumerate(vecs[mask]):
            exp[np.where(mask)[0][i]] = _REFERENCE_MAP[tuple(row)]
        if not np.array_equal(got, exp):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
