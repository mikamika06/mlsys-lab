def _oracle(accesses):
    result = []
    for warp in accesses:
        counts = [0] * 32
        for address in warp:
            bank = int(address) % 32
            counts[bank] += 1
        result.append(max(counts))
    return result


def grade(sol, fx) -> dict:
    cases = [
        [list(range(32))],
        [[0] * 32],
        [[i * 32 for i in range(32)]],
        [[i * 2 for i in range(32)]],
        [
            list(range(32)),
            [5 + i * 32 for i in range(32)],
            [i % 7 for i in range(32)],
        ],
    ]

    ok = 1.0
    for accesses in cases:
        try:
            got = list(sol.bank_conflict_degree([list(w) for w in accesses]))
        except Exception:
            ok = 0.0
            break

        expected = _oracle(accesses)
        if got != expected:
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
