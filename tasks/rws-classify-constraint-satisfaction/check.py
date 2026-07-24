import numpy as np


def _oracle(mask, L, H, d_ff, target_heads, target_ff):
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (L, H + d_ff):
        return False
    heads = np.sum(arr[:, :H], axis=1)
    ff = np.sum(arr[:, H:], axis=1)
    return bool(np.all(heads == target_heads) and np.all(ff == target_ff))


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(7)

    configs = [
        (2, 4, 6, 2, 3),
        (3, 8, 5, 4, 2),
        (1, 3, 7, 1, 5),
        (4, 6, 4, 6, 1),
    ]

    for L, H, d_ff, target_heads, target_ff in configs:
        for _ in range(5):
            mask = rng.integers(0, 2, size=(L, H + d_ff), dtype=np.int8).astype(bool)
            cases.append((mask, L, H, d_ff, target_heads, target_ff))

    ok = 1.0
    for mask, L, H, d_ff, target_heads, target_ff in cases:
        expected = _oracle(mask, L, H, d_ff, target_heads, target_ff)
        try:
            got = sol.classify_mask(
                mask,
                L,
                H,
                d_ff,
                target_heads,
                target_ff,
            )
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, (bool, np.bool_)) or bool(got) != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
