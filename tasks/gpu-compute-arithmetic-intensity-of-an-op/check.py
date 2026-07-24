def _ref(flops, read_bytes, write_bytes=0):
    total = read_bytes + write_bytes
    if total == 0:
        return float('inf')
    ai = flops / total
    return round(ai, 6)

def grade(sol, fx) -> dict:
    cases = [
        (500, 100, 200),
        (123456, 1111, 2222),
        (10, 5, 5),
        (0, 0, 0),
        (42, 0, 0)
    ]
    ok = 1.0
    for flops, read, write in cases:
        try:
            got = sol.arithmetic_intensity(flops, read, write)
        except Exception:
            ok = 0.0
            break
        expected = _ref(flops, read, write)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
