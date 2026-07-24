import sys


def _ref(values):
    width = sys.hash_info.width
    mask = (1 << width) - 1

    def shuffle_bits(h):
        return (((h ^ 89869747) ^ (h << 16)) * 3644798167) & mask

    unique = set(values)
    h = 0
    for value in unique:
        h ^= shuffle_bits(hash(value))

    h ^= ((len(unique) + 1) * 1927868237) & mask
    h ^= h >> 11
    h ^= h >> 25
    h = (h * 69069 + 907133923) & mask

    if h == mask:
        h = 590923713

    if h >= (1 << (width - 1)):
        h -= 1 << width
    return h


def grade(sol, fx) -> dict:
    cases = [
        [],
        [0],
        [1, 2, 3],
        [3, 2, 1],
        [-1, 0, 1, 2, 1000],
        list(range(20)),
        [5, 5, 5, 6, 7],
        [-10, 50, 100, 200, 300, 400],
    ]

    ok = 1.0
    for values in cases:
        try:
            got = sol.frozenset_hash(list(values))
            expected = _ref(list(values))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
