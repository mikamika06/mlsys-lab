def _oracle(layer_bytes, gpu_cap, cpu_cap):
    n = len(layer_bytes)
    states = {(0, 0): []}
    for idx, size in enumerate(layer_bytes):
        nxt = {}
        for (g, c), chosen in states.items():
            if (g, c) not in nxt:
                nxt[(g, c)] = chosen + [2]
            if g + size <= gpu_cap:
                key = (g + size, c)
                if key not in nxt:
                    nxt[key] = chosen + [0]
            if c + size <= cpu_cap:
                key = (g, c + size)
                if key not in nxt:
                    nxt[key] = chosen + [1]
        states = nxt
    best = None
    best_resident = -1
    for (g, c), chosen in states.items():
        resident = g + c
        if resident > best_resident:
            best_resident = resident
            best = chosen
    return best


def _disk_bytes(layer_bytes, placement):
    return sum(size for size, loc in zip(layer_bytes, placement) if loc == 2)


def grade(sol, fx) -> dict:
    cases = [
        ([8, 5, 4], 9, 5),
        ([7, 7, 7, 2], 8, 9),
        ([3, 6, 10, 4, 5], 12, 10),
        ([11, 1, 8, 2, 9, 3], 13, 14),
        ([4, 4, 4, 4, 4], 6, 8),
        ([6, 13, 5, 9, 2, 7, 3], 15, 16),
    ]
    ok = 1.0
    for layers, gpu, cpu in cases:
        try:
            got = list(sol.split_layers(list(layers), gpu, cpu))
        except Exception:
            ok = 0.0
            break
        if len(got) != len(layers):
            ok = 0.0
            break
        if any(x not in (0, 1, 2) for x in got):
            ok = 0.0
            break
        g = sum(s for s, x in zip(layers, got) if x == 0)
        c = sum(s for s, x in zip(layers, got) if x == 1)
        if g > gpu or c > cpu:
            ok = 0.0
            break
        ref = _oracle(layers, gpu, cpu)
        if _disk_bytes(layers, got) != _disk_bytes(layers, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
