def evaluate_throughput(fused_ms, unfused_ms):
    if not fused_ms or not unfused_ms:
        return 0.0
    ratios = [u / f if f > 0 else 0.0 for f, u in zip(fused_ms, unfused_ms)]
    return sum(ratios) / len(ratios)
