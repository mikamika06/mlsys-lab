def _oracle_probe_sequence(hash_value, size, steps):
    mask = size - 1
    perturb = hash_value
    slot = hash_value & mask
    result = []

    for _ in range(steps):
        result.append(slot)
        slot = (5 * slot + 1 + perturb) & mask
        perturb >>= 5

    return result


def grade(sol, fx) -> dict:
    cases = [
        (hash(0), 8, 1),
        (hash(1), 16, 6),
        (hash(42), 8, 5),
        (hash(13), 32, 10),
        (hash(-7), 64, 9),
        (hash(2**20 + 3), 128, 12),
    ]

    exact = 1.0

    for hash_value, size, steps in cases:
        expected = _oracle_probe_sequence(hash_value, size, steps)

        try:
            actual = sol.dict_probe_sequence(hash_value, size, steps)
        except Exception:
            exact = 0.0
            break

        if actual != expected:
            exact = 0.0
            break

    return {"exact_match": exact}
