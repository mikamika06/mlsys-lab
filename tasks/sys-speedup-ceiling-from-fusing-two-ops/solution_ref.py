def fused_elementwise_speedup(x: list[float], y: list[float]) -> float:
    z = [xi * yi for xi, yi in zip(x, y)]
    out = [zi + 1.0 for zi in z]
    itemsize = 8
    unfused_bytes = len(x) * itemsize + len(y) * itemsize + len(z) * itemsize + len(z) * itemsize + len(out) * itemsize
    fused_bytes = len(x) * itemsize + len(y) * itemsize + len(out) * itemsize
    return float(unfused_bytes / fused_bytes)
