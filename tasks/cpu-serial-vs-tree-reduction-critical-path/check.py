def _reference(n, latency):
    import math
    serial = (n - 1) * latency
    tree = math.ceil(math.log2(n)) * latency
    return serial, tree

def grade(sol, fx) -> dict:
    cases = [
        (5, 2),
        (8, 3),
        (7, 5),
        (16, 1),
        (13, 4)
    ]
    ok = 1.0
    for n, latency in cases:
        try:
            got = sol.critical_path_lengths(n, latency)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            ref = _reference(n, latency)
            if got != ref:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
