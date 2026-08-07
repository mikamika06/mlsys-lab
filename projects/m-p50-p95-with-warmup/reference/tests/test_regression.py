import sys

sys.path.insert(0, ".")
from measure import benchmark

class Clock:
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

def test_outlier_rejection_drops_high_values():
    delays = [10.0, 1.0, 1.0] + [1.0, 1.1, 100.0, 1.2, 0.9, 100.0, 1.0, 1.1, 1.0, 0.9]
    clock = Clock(delays)
    def fn():
        clock.advance()

    res = benchmark(fn, 3, 10, reject_outliers=True, clock=clock)
    assert res["p95"] < 10.0, f"Outliers were not rejected, p95 is {res['p95']}"
