def detect_missing_warmup(latencies, threshold_factor):
    if not latencies:
        return False
    first = latencies[0]
    rest_mean = sum(latencies[1:]) / len(latencies[1:]) if len(latencies) > 1 else first
    return first > rest_mean * threshold_factor
