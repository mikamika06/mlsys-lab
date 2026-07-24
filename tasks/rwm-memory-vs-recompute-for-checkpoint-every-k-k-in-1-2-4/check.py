def _oracle(L, activation_bytes):
    def recompute_for_k(k):
        checkpoints = list(range(0, L + 1, k))
        if checkpoints[-1] != L:
            checkpoints.append(L)

        stored = (len(checkpoints)) * activation_bytes

        extra = 0
        for start, end in zip(checkpoints[:-1], checkpoints[1:]):
            segment_length = end - start
            extra += max(0, segment_length - 1)

        return [k, stored, extra]

    return [recompute_for_k(k) for k in (1, 2, 4)]


def grade(sol, fx) -> dict:
    cases = [
        (1, 64),
        (2, 128),
        (5, 1024),
        (7, 256),
        (16, 4096),
        (33, 512),
    ]

    ok = 1.0
    for L, activation_bytes in cases:
        try:
            got = sol.checkpoint_curve(L, activation_bytes)
        except Exception:
            ok = 0.0
            break

        if got != _oracle(L, activation_bytes):
            ok = 0.0
            break

    return {"exact_match": ok}
