import numpy as np

SCENARIOS = [
    {"num_cores": 4, "max_streams": 12, "scaling": 120.0, "penalty": 0.4},
    {"num_cores": 8, "max_streams": 16, "scaling": 80.0, "penalty": 0.3},
    {"num_cores": 16, "max_streams": 32, "scaling": 50.0, "penalty": 0.5},
    {"num_cores": 2, "max_streams": 8, "scaling": 200.0, "penalty": 0.6},
]


def make_bench_fn(num_cores, scaling, penalty):
    def bench(streams):
        if streams <= num_cores:
            return float(streams * scaling)
        decay = 1.0 + penalty * (streams - num_cores) ** 1.2
        return float((num_cores * scaling) / decay)
    return bench


def reference_sweep(bench_fn, max_streams):
    results = {}
    for n in range(1, max_streams + 1):
        results[n] = float(bench_fn(n))

    knee_point = 1
    max_tp = 0.0
    for n in sorted(results.keys()):
        if results[n] > max_tp:
            max_tp = results[n]
            knee_point = n

    return results, knee_point
