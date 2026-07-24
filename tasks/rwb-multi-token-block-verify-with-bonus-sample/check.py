def _sample(dist, u):
    total = sum(dist)
    if total <= 0:
        return 0
    acc = 0.0
    for i, x in enumerate(dist):
        acc += x / total
        if u < acc:
            return i
    return len(dist) - 1


def _ref(draft, p, q, rng):
    accepted = 0
    rng_i = 0
    rejected = False
    for i, token in enumerate(draft):
        prob = 0.0
        if q[i][token] > 0:
            prob = min(1.0, p[i][token] / q[i][token])
        if rng[rng_i] < prob:
            rng_i += 1
            accepted += 1
        else:
            rng_i += 1
            rejected = True
            break

    if rejected:
        i = accepted
        residual = [max(p[i][j] - q[i][j], 0.0) for j in range(len(p[i]))]
        bonus = _sample(residual, rng[rng_i])
    else:
        bonus = _sample(p[-1], rng[rng_i])

    return accepted, draft[:accepted] + [bonus]


def grade(sol, fx) -> dict:
    cases = [
        (
            [2, 1],
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]],
            [[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]],
            [0.1, 0.8, 0.2],
        ),
        (
            [0, 2, 1],
            [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]],
            [[0.5, 0.4, 0.1], [0.4, 0.2, 0.4], [0.2, 0.7, 0.1]],
            [0.9, 0.05, 0.4, 0.7],
        ),
        (
            [1, 1, 0],
            [[0.2, 0.8], [0.2, 0.8], [0.9, 0.1]],
            [[0.2, 0.8], [0.3, 0.7], [0.8, 0.2]],
            [0.2, 0.3, 0.4, 0.1],
        ),
        (
            [3],
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.25, 0.25, 0.25, 0.25]],
            [0.0, 0.75],
        ),
    ]

    ok = 1.0
    for draft, p, q, rng in cases:
        expected = _ref(draft, p, q, rng)
        try:
            got = sol.verify_block(draft, p, q, rng)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
