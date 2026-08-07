import time


def measure_compiled_speedup(fn_a, fn_b, sample_inputs, warmup_runs=3, timed_runs=50):
    for _ in range(warmup_runs):
        for x in sample_inputs:
            fn_a(x)
            fn_b(x)

    lat_a = []
    lat_b = []
    for x in sample_inputs:
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            fn_a(x)
            t1 = time.perf_counter()
            lat_a.append(t1 - t0)

            t2 = time.perf_counter()
            fn_b(x)
            t3 = time.perf_counter()
            lat_b.append(t3 - t2)

    return lat_a, lat_b


def run_paired_benchmarks(fn_a, fn_b, sample_inputs, warmup_runs=3, timed_runs=50):
    return measure_compiled_speedup(fn_a, fn_b, sample_inputs, warmup_runs, timed_runs)
