import time
import numpy as np

def benchmark(fn, warmup_iters, measure_iters, percentiles):
    """
    Calls `fn` for `warmup_iters` without recording.
    Then calls `fn` for `measure_iters`, recording each duration in nanoseconds using time.perf_counter_ns().
    Returns a dict mapping each percentile in `percentiles` to the computed percentile value.
    Use numpy.percentile with method='linear' (default).
    """
    raise NotImplementedError

def find_stable_iters(fn, target_rel_err, start_iters=10, max_iters=10000):
    """
    Finds the minimum iterations needed for the p90 latency to stabilize.

    Starts with `iters = start_iters`.
    In a loop:
      1. a = benchmark(fn, 10, iters, [90])[90]
      2. b = benchmark(fn, 0, iters, [90])[90]
      3. Compute rel_err = abs(a - b) / max(a, b) (if max(a, b) > 0, else 0.0)
      4. If rel_err <= target_rel_err, return iters
      5. iters *= 2

    Returns max_iters if iters >= max_iters before stabilizing.
    """
    raise NotImplementedError
