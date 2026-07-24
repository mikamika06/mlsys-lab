def _ref(m, k, n):
    x = m * k
    w = k * n
    b = n
    out = m * n

    unfused = (
        x +
        w +
        b +
        out +
        out +
        b +
        out +
        out +
        out
    )

    fused = x + w + b + out

    return {"unfused": unfused, "fused": fused}


def grade(sol, fx) -> dict:
    cases = [
        (1, 1, 1),
        (4, 8, 16),
        (32, 64, 128),
        (7, 13, 5),
        (256, 256, 64),
    ]

    ok = 1.0
    for m, k, n in cases:
        try:
            got = sol.count_hbm_round_trips(m, k, n)
        except Exception:
            ok = 0.0
            break

        if got != _ref(m, k, n):
            ok = 0.0
            break

    return {"modeled_mem_access": ok}
