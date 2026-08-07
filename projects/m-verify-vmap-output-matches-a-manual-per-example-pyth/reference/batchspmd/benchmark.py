import time
from batchspmd.vmap import per_example_loop


def benchmark_vmap_speedup(fn_single, fn_batched, x_batches, axis=0, timer=None):
    """Measures execution speedup of fn_batched over per_example_loop for a list of x_batches."""
    if timer is None:
        timer = time.perf_counter
    results = {}
    for x in x_batches:
        b_size = x.shape[axis]
        t0 = timer()
        _ = per_example_loop(fn_single, x, axis=axis)
        t1 = timer()
        t_loop = max(t1 - t0, 1e-9)

        t2 = timer()
        _ = fn_batched(x)
        t3 = timer()
        t_vmap = max(t3 - t2, 1e-9)

        speedup = t_loop / t_vmap
        results[b_size] = {
            "loop_time": float(t_loop),
            "vmap_time": float(t_vmap),
            "speedup": float(speedup),
        }
    return results
