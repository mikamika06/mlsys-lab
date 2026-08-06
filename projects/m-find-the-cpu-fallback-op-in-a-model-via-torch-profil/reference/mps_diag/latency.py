def measure_latency_cliff(fallback_latency, native_latency):
    if native_latency == 0:
        return float("inf")
    return float(fallback_latency) / float(native_latency)
