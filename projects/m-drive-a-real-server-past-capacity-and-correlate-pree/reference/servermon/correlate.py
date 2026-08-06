def compute_p99_correlation(preemptions, latencies):
    n = len(preemptions)
    if n == 0:
        return {"correlation": 0.0, "p99": 0.0}
    mean_p = sum(preemptions) / n
    mean_l = sum(latencies) / n
    num = sum((preemptions[i] - mean_p) * (latencies[i] - mean_l) for i in range(n))
    den_p = sum((preemptions[i] - mean_p) ** 2 for i in range(n))
    den_l = sum((latencies[i] - mean_l) ** 2 for i in range(n))
    if den_p == 0 or den_l == 0:
        corr = 0.0
    else:
        corr = num / ((den_p * den_l) ** 0.5)

    sorted_lat = sorted(latencies)
    idx = int(0.99 * (n - 1))
    p99 = sorted_lat[idx]
    return {"correlation": round(corr, 4), "p99": p99}
