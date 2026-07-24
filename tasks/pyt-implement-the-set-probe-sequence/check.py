def _oracle(key, mask):
    # Integer hashes are the CPython oracle for these fixtures.
    h = hash(int(key))
    perturb = h
    i = h & mask
    out = []
    for _ in range(9):
        out.append(i)
        i = (i + 1) & mask
    while len(out) < 20:
        i = (i * 5 + 1 + perturb) & mask
        out.append(i)
        perturb >>= 5
    return out


def grade(sol, fx) -> dict:
    cases = [
        (0, 7),
        (1, 7),
        (5, 7),
        (13, 15),
        (123456, 31),
        (-9, 63),
    ]
    ok = 1.0
    for key, mask in cases:
        try:
            got = sol.set_probe_sequence(key, mask)
        except Exception:
            ok = 0.0
            break
        if list(got) != _oracle(key, mask):
            ok = 0.0
            break
    return {"exact_match": ok}
