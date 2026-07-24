import numpy as np


def _key(x):
    bits = np.asarray(x, dtype=np.float64).view(np.uint64)
    return np.where((bits >> np.uint64(63)) != 0,
                    bits ^ np.uint64(0xFFFFFFFFFFFFFFFF),
                    bits ^ np.uint64(0x8000000000000000))


def _oracle(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    bits = values.view(np.uint64)
    order = np.argsort(_key(values), kind="stable")
    ordered = values[order]

    nan_mask = np.isnan(ordered)
    if np.any(nan_mask):
        return ordered[np.flatnonzero(nan_mask)[0]]

    current = list(ordered)
    while len(current) > 1:
        nxt = []
        i = 0
        while i + 1 < len(current):
            nxt.append(np.float64(current[i] + current[i + 1]))
            i += 2
        if i < len(current):
            nxt.append(current[i])
        current = nxt
    return np.float64(current[0])


def _same_bits(a, b):
    return np.asarray(a, dtype=np.float64).view(np.uint64).item() == np.asarray(b, dtype=np.float64).view(np.uint64).item()


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        np.array([1e20, 1.0, -1e20, 3.0], dtype=np.float64),
        np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float64),
        np.array([np.inf, -np.inf, 5.0], dtype=np.float64),
        np.array([np.nan, 1.0, -2.0], dtype=np.float64),
    ]
    for _ in range(8):
        cases.append(rng.normal(size=17).astype(np.float64))

    ok = 1.0
    for case in cases:
        expected = _oracle(case)
        first = None
        for perm in [case] + [case[rng.permutation(len(case))] for _ in range(12)]:
            try:
                got = sol.stable_sum(perm)
            except Exception:
                return {"exact_match": 0.0}
            if not _same_bits(got, expected):
                ok = 0.0
                break
            if first is None:
                first = np.asarray(got, dtype=np.float64).view(np.uint64).item()
            elif np.asarray(got, dtype=np.float64).view(np.uint64).item() != first:
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"exact_match": ok}
