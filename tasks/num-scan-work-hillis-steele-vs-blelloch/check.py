def _ref(n, algorithm):
    if n <= 0:
        raise ValueError("non-positive size")
    if algorithm == "hillis_steele":
        count = 0
        d = 1
        while d < n:
            for i in range(d, n):
                count += 1
            d *= 2
        return count
    if algorithm == "blelloch":
        m = 1
        while m < n:
            m *= 2
        count = 0
        stride = 2
        while stride <= m:
            count += m // stride
            stride *= 2
        stride = m
        while stride >= 2:
            count += m // stride
            stride //= 2
        return count
    raise ValueError("bad algorithm")


def grade(sol, fx) -> dict:
    cases = [
        (1, "hillis_steele"),
        (2, "hillis_steele"),
        (5, "hillis_steele"),
        (8, "hillis_steele"),
        (9, "hillis_steele"),
        (1, "blelloch"),
        (3, "blelloch"),
        (8, "blelloch"),
        (10, "blelloch"),
        (17, "blelloch"),
    ]
    ok = 1.0
    for n, algorithm in cases:
        try:
            expected = _ref(n, algorithm)
            got = sol.scan_work(n, algorithm)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"modeled_access_count": ok}
