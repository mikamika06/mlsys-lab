def arithmetic_intensity(m: int, n: int, k: int, itemsize: int = 2) -> float:
    """Calculate arithmetic intensity in FLOPs per byte."""
    flops = 2 * m * n * k
    total_bytes = (m * k + k * n + m * n) * itemsize
    return flops / total_bytes


def attainable_gflops(ai: float, peak_gflops: float, peak_gbps: float = 546.0) -> float:
    """Calculate attainable performance in GFLOP/s based on roofline bound."""
    return min(peak_gflops, ai * peak_gbps)


def fit_empirical_roofline(profile_data: list[dict], peak_gbps: float = 546.0) -> dict:
    """Fit empirical roofline model on profiled matrix multiplication runs."""
    profiles = []
    max_gflops = 0.0
    max_bw = 0.0

    for run in profile_data:
        m, n, k = run["m"], run["n"], run["k"]
        sec = run["elapsed_sec"]
        itemsize = run.get("itemsize", 2)

        flops = 2 * m * n * k
        total_bytes = (m * k + k * n + m * n) * itemsize

        ai = flops / total_bytes
        gflops = (flops / 1e9) / sec
        gbps = (total_bytes / 1e9) / sec

        if gflops > max_gflops:
            max_gflops = gflops
        if gbps > max_bw:
            max_bw = gbps

        profiles.append({
            "m": m,
            "n": n,
            "k": k,
            "ai": ai,
            "achieved_gflops": gflops,
            "achieved_gbps": gbps,
        })

    knee_ai = max_gflops / peak_gbps if peak_gbps > 0 else 0.0

    for p in profiles:
        p["is_memory_bound"] = p["ai"] < knee_ai

    return {
        "peak_bandwidth_gbps": peak_gbps,
        "empirical_peak_gflops": max_gflops,
        "knee_ai": knee_ai,
        "max_achieved_bw_gbps": max_bw,
        "max_bw_utilization_pct": (max_bw / peak_gbps) * 100.0 if peak_gbps > 0 else 0.0,
        "profiles": profiles,
    }
