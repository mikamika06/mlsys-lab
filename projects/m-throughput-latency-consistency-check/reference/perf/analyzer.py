def compute_metrics(events):
    if not events:
        return 0.0, 0.0, 0.0, 0.0

    t_start = min(e["host_start"] for e in events)
    t_end = max(e["device_end"] for e in events)
    throughput = len(events) / (t_end - t_start)

    h_lat = sum(e["host_end"] - e["host_start"] for e in events) / len(events)
    d_lat = sum(e["device_end"] - e["device_start"] for e in events) / len(events)
    e2e_lat = sum(e["device_end"] - e["host_start"] for e in events) / len(events)

    return throughput, h_lat, d_lat, e2e_lat

def consistency_error(throughput, mean_latency, concurrency):
    if concurrency <= 0:
        return 0.0
    return abs(throughput * mean_latency - concurrency) / concurrency

def validate_trace(events, concurrency, tol=0.05):
    t, h, d, e2e = compute_metrics(events)
    err = consistency_error(t, e2e, concurrency)
    if err > tol:
        raise ValueError()
    return err
