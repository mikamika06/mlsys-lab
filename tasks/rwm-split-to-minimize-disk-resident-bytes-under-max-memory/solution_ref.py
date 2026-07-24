def split_layers(layer_bytes, gpu_cap, cpu_cap):
    states = {(0, 0): []}
    for size in layer_bytes:
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
        if g + c > best_resident:
            best_resident = g + c
            best = chosen
    return best
