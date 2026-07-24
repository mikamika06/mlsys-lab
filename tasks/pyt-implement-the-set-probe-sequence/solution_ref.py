def set_probe_sequence(key: int, mask: int) -> list[int]:
    perturb = key
    i = key & mask
    out = []

    for _ in range(9):
        out.append(i)
        i = (i + 1) & mask

    while len(out) < 20:
        i = (i * 5 + 1 + perturb) & mask
        out.append(i)
        perturb >>= 5

    return out
