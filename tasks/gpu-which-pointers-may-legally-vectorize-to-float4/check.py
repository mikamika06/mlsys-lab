def grade(sol, fx) -> dict:
    # Generate a deterministic mix of aligned and unaligned addresses
    addrs = []
    for i in range(20):
        addrs.append(i * 16)       # all aligned
    for i in range(1, 20):
        addrs.append(i * 16 + 4)   # offset by 4 bytes -> not aligned
    for offset in [0, 1, 2, 3, 4, 8, 12, 15, 16, 32, 48, 64, 100, 128, 256]:
        addrs.append(offset)

    ref = [addr % 16 == 0 for addr in addrs]

    try:
        got = sol.vectorizable_pointers(addrs)
    except Exception:
        return {"exact_match": 0.0}

    if len(got) != len(ref):
        return {"exact_match": 0.0}

    matches = sum(1 for g, r in zip(got, ref) if bool(g) == bool(r))
    exact_match = matches / len(ref)
    return {"exact_match": float(exact_match)}
