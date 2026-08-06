"""Thread scaling and oversubscription calculations."""


def find_oversubscription_point(topology, latency_data):
    """Derive oversubscription thread count given core topology and latency sweep data."""
    p_cores = topology.get("p_cores", 0)
    for threads in sorted(latency_data.keys()):
        if threads > p_cores:
            return threads
    return p_cores + 1


def analyze_thread_sweep(latency_data, work_units):
    """Compute throughput (ops/sec) and relative latency scaling for a thread count sweep."""
    base_latency = latency_data[1]
    result = {}
    for threads, latency_ms in sorted(latency_data.items()):
        throughput = work_units / (latency_ms / 1000.0)
        latency_ratio = latency_ms / base_latency
        result[threads] = {
            "throughput": throughput,
            "latency_ratio": latency_ratio
        }
    return result
