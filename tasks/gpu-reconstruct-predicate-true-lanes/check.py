def _ref(mask):
    lanes = []
    for i in range(32):
        if mask & (1 << i):
            lanes.append(i)
    return lanes

def grade(sol, fx) -> dict:
    cases = [
        0,
        1,
        2**5 | 2**12 | 2**31,
        0xffffffff,
        0x80000000
    ]
    ok = 1.0
    for mask in cases:
        try:
            got = sol.reconstruct_lanes(mask)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, (list, tuple)):
            ok = 0.0
            break
        if list(got) != _ref(mask):
            ok = 0.0
            break
    return {"exact_match": ok}
