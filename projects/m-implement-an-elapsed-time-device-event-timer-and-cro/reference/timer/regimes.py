def classify_regimes(profiles):
    ranked = []
    for p in profiles:
        k = p["kernel_ms"]
        h = p["host_ms"]
        if k > h * 2:
            rid = 0
        elif h > k * 2 and k < 0.1:
            rid = 3
        elif h > k:
            rid = 1
        else:
            rid = 2
        ranked.append({"name": p["name"], "regime_id": rid})
    return ranked
