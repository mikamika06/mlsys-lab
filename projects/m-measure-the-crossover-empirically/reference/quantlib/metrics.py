def compute_ratio(perf_a, perf_b):
    if perf_b == 0:
        return 0.0
    return float(perf_a) / float(perf_b)
