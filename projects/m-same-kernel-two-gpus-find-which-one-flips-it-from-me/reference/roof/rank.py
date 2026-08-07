def rank_device_kernel_pairs(pairs):
    scored = []
    for p in pairs:
        ridge = p["peak_flop"] / p["peak_bw"]
        intensity = p["flops"] / p["bytes"]
        scored.append((intensity - ridge, p))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]
