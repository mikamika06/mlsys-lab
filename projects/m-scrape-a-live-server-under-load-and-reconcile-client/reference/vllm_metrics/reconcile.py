def interpolate_percentile(histogram, percentile):
    if not histogram:
        return 0.0
    total = histogram[-1][1]
    if total <= 0:
        return 0.0
    target = percentile * total
    prev_le = 0.0
    prev_count = 0.0
    for le, count in histogram:
        if count >= target:
            if count == prev_count:
                return le
            fraction = (target - prev_count) / (count - prev_count)
            return prev_le + fraction * (le - prev_le)
        prev_le = le
        prev_count = count
    return histogram[-1][0]


def reconcile_latency(client_latencies, server_histogram):
    if not client_latencies:
        return {"server_p99": 0.0, "client_p99": 0.0, "rel_err": 0.0}
    sorted_client = sorted(client_latencies)
    idx = int(0.99 * len(sorted_client))
    client_p99 = sorted_client[min(idx, len(sorted_client) - 1)]
    server_p99 = interpolate_percentile(server_histogram, 0.99)
    if server_p99 <= 0:
        rel_err = 1.0
    else:
        rel_err = abs(client_p99 - server_p99) / server_p99
    return {
        "server_p99": server_p99,
        "client_p99": client_p99,
        "rel_err": rel_err
    }
