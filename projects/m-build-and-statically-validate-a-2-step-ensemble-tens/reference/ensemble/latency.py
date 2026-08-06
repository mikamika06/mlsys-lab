def compute_ensemble_latency(step1_latencies, step2_latencies):
    s1 = sorted(step1_latencies)
    s2 = sorted(step2_latencies)
    combined = sorted([a + b for a, b in zip(s1, s2)])
    n = len(combined)
    p50 = combined[n // 2]
    p99 = combined[int(0.99 * (n - 1))]
    return {"p50": float(p50), "p99": float(p99)}
