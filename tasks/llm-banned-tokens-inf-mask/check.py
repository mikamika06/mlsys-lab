import numpy as np

def _reference(logits, banned):
    out = logits.copy()
    if out.ndim == 1:
        out[banned] = -np.inf
    else:
        out[:, banned] = -np.inf
    return out

def grade(sol, fx) -> dict:
    cases = [
        ([0.5, -1.2, 3.4], [1, 2]),
        ([[0.5, -1.2, 3.4], [1.0, 0.0, -0.5]], [1, 2]),
        ([1.0, 2.0, 3.0], []),
        ([[1.0, 2.0], [3.0, 4.0]], [0]),
    ]
    ok = 1.0
    for logits, banned in cases:
        try:
            got = sol.mask_banned_tokens(logits, banned)
        except Exception:
            return {"exact_match": 0.0}
        expected = _reference(np.array(logits), banned)
        if not np.array_equal(np.array(got), expected):
            ok = 0.0
            break
    return {"exact_match": ok}
