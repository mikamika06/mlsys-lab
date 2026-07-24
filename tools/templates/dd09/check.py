def _ref(keys, size):
    mask = size - 1
    entries = []
    indices = [-1] * size
    for key in keys:
        perturb = key
        i = key & mask
        while indices[i] != -1 and entries[indices[i]] != key:
            i = (i * 5 + perturb + 1) & mask
            perturb >>= 5
        if indices[i] == -1:
            entries.append(key)
            indices[i] = len(entries) - 1
        # else: existing key -> update, entries/indices unchanged
    return [entries, indices]


def grade(sol, fx) -> dict:
    cases = [
        ([1, 9, 17], 8),           # all collide at slot 1 -> perturb chain
        ([0, 1, 2, 3], 8),         # no collisions
        ([5, 13, 21, 29, 5], 8),   # collisions + a duplicate update
        ([3, 11, 19, 27, 8], 16),
        (list(range(10)), 32),
    ]
    ok = 1.0
    for keys, size in cases:
        try:
            got = sol.compact_insert(list(keys), size)
            got = [list(got[0]), list(got[1])]
        except Exception:
            ok = 0.0
            break
        if got != _ref(list(keys), size):
            ok = 0.0
            break
    return {"exact_match": ok}
