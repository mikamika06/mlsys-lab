def measure_throughput(concurrency_levels):
    throughputs = []
    for c in concurrency_levels:
        if c <= 0:
            throughputs.append(0.0)
        else:
            base = c * 15.0
            penalty = max(0.0, (c - 16.0) * 0.8)
            throughputs.append(max(1.0, base - penalty))
    return throughputs
