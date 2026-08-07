import time


def measure_overhead(broken_fn, fixed_fn, sample_args, num_iters=100):
    for _ in range(5):
        broken_fn(*sample_args)
        fixed_fn(*sample_args)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        broken_fn(*sample_args)
    t1 = time.perf_counter()
    t_broken = t1 - t0

    t0 = time.perf_counter()
    for _ in range(num_iters):
        fixed_fn(*sample_args)
    t1 = time.perf_counter()
    t_fixed = t1 - t0
    return {
        "broken_time": t_broken,
        "fixed_time": t_fixed,
        "ratio": t_broken / max(t_fixed, 1e-9),
    }
