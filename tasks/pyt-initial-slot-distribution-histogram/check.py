def _oracle(keys, size):
    counts = [0] * size
    mask = size - 1
    for key in keys:
        slot = hash(key) & mask
        counts[slot] += 1
    return counts


def grade(sol, fx) -> dict:
    cases = [
        ([1, 9, 17, 2], 8),
        ([0, 8, 16, 24, 3, 11], 8),
        (list(range(32)), 16),
        ([123456789, -1, -9, 42, 42], 32),
        ([-64, -32, 0, 32, 64, 96], 8),
    ]
    ok = 1.0
    for keys, size in cases:
        try:
            got = list(sol.initial_slot_histogram(list(keys), size))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(keys, size):
            ok = 0.0
            break
    return {"exact_match": ok}
