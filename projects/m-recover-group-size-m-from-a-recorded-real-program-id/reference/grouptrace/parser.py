def parse_trace(trace):
    max_m = max(p[0] for p in trace)
    max_n = max(p[1] for p in trace)
    return max_m + 1, max_n + 1
