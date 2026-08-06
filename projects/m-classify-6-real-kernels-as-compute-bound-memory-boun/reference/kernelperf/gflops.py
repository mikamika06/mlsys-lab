def compute_achieved_gflops(compute_pct, peak_tflops):
    return (compute_pct / 100.0) * peak_tflops * 1000.0


def rank_kernels_by_gflops(kernels, peak_tflops):
    scored = []
    for k in kernels:
        gfs = compute_achieved_gflops(k.get("compute_pct", 0.0), peak_tflops)
        scored.append((k["name"], gfs))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]
