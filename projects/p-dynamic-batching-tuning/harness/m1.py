def check(workdir):
    m = {"profile_computed": 0.0, "monotonic_latency": 0.0}
    try:
        from batching.profiler import measure_latency_curve
    except Exception:
        return m

    sizes = [1, 2, 4, 8, 16, 32]
    try:
        res = measure_latency_curve(sizes)
    except Exception:
        return m

    if isinstance(res, dict) and len(res) == len(sizes):
        m["profile_computed"] = 1.0
        latencies = [res[s] for s in sorted(res.keys())]
        if all(latencies[i] <= latencies[i+1] for i in range(len(latencies)-1)):
            m["monotonic_latency"] = 1.0
    return m
