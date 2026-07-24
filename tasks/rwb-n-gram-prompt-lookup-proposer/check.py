import numpy as np


def _oracle_propose(context, lo, hi, k):
    context = np.asarray(context)
    n = len(context)
    for L in range(hi, lo - 1, -1):
        if L <= 0 or L > n:
            continue
        suffix = context[n - L:n]
        best_i = -1
        max_i = n - 2 * L
        for i in range(0, max_i + 1):
            if np.array_equal(context[i:i + L], suffix):
                best_i = i  # keep scanning forward -> ends on the rightmost match
        if best_i >= 0:
            start = best_i + L
            return context[start:start + k]
    return np.array([], dtype=context.dtype)


def _cases(sequence):
    n = len(sequence)
    out = []
    for T in [15, 18, 33, 39, 50, 55, n - 2, n]:
        T = max(1, min(T, n))
        for lo, hi, k in [(2, 5, 3), (1, 3, 2), (3, 6, 4)]:
            out.append((sequence[:T], lo, hi, k))
    return out


def grade(sol, fx) -> dict:
    sequence = np.asarray(fx["sequence"], dtype=np.int64)

    ok = 1.0
    for context, lo, hi, k in _cases(sequence):
        expected = _oracle_propose(context, lo, hi, k)
        try:
            got = np.asarray(sol.propose_tokens(context.copy(), lo, hi, k), dtype=np.int64)
        except Exception:
            ok = 0.0
            break
        if got.shape != expected.shape or not np.array_equal(got, expected):
            ok = 0.0
            break

    return {"exact_match": ok}
