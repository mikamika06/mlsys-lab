import numpy as np

def benchmark(fn, warmup, iters, reject_outliers=False, clock=None):
    warmup_times = []
    for _ in range(warmup):
        t0 = clock()
        fn()
        t1 = clock()
        warmup_times.append(t1 - t0)

    measure_times = []
    for _ in range(iters):
        t0 = clock()
        fn()
        t1 = clock()
        measure_times.append(t1 - t0)

    if reject_outliers and len(measure_times) > 0:
        q1 = np.percentile(measure_times, 25)
        q3 = np.percentile(measure_times, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        measure_times = [t for t in measure_times if t <= upper]

    if not measure_times:
        return {"p50": 0.0, "p95": 0.0, "cold_start_ratio": 0.0}

    p50 = np.percentile(measure_times, 50)
    p95 = np.percentile(measure_times, 95)

    if warmup > 0 and p50 > 0:
        cold = warmup_times[0] / p50
    else:
        cold = 0.0

    return {
        "p50": float(p50),
        "p95": float(p95),
        "cold_start_ratio": float(cold)
    }

class MockClock:
    def __init__(self, delays):
        self.delays = delays
        self.t = 0.0
        self.i = 0
    def __call__(self):
        return self.t
    def advance(self):
        if self.i < len(self.delays):
            self.t += self.delays[self.i]
            self.i += 1

def make_clock_and_fn(delays):
    clock = MockClock(delays)
    def fn():
        clock.advance()
    return clock, fn

DELAYS_M1 = [100.0, 2.0, 2.0] + [2.0, 2.1, 2.0, 2.2, 1.9, 2.0, 2.3, 2.1, 2.0, 1.9]
DELAYS_M2 = [100.0, 2.0, 2.0] + [2.0, 2.1, 50.0, 2.2, 1.9, 60.0, 2.3, 2.1, 2.0, 1.9]
