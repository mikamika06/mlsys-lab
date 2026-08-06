def run_sweep(weights, sizes):
    results = {}
    for sz in sizes:
        results[sz] = 1.0 / (sz ** 0.5)
    return results
