def compute_robust_stats(latencies):
    if not latencies:
        return {"median": 0.0, "mad": 0.0, "p95": 0.0}
    s = sorted(latencies)
    n = len(s)
    if n % 2 == 1:
        med = float(s[n // 2])
    else:
        med = float(s[n // 2 - 1] + s[n // 2]) / 2.0

    devs = sorted([abs(x - med) for x in latencies])
    if n % 2 == 1:
        mad = float(devs[n // 2])
    else:
        mad = float(devs[n // 2 - 1] + devs[n // 2]) / 2.0

    p95_idx = int(0.95 * (n - 1))
    p95 = float(s[p95_idx])
    return {"median": med, "mad": mad, "p95": p95}
