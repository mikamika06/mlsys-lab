def count_hbm_round_trips(m: int, k: int, n: int) -> dict:
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

    return {
        "unfused": unfused,
        "fused": fused,
    }
