import sys

sys.path.insert(0, ".")
from bakeoff.benchmark import Benchmark


def test_run_stable_returns_median():
    class MockClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def advance(self, dt):
            self.t += dt

    class FlakyEngine:
        def __init__(self, clock):
            self.clock = clock
            self.calls = 0

        def setup(self, c, b, q):
            pass

        def prefill(self, t):
            times = [10.0, 50.0, 20.0]
            self.clock.advance(times[self.calls % 3])
            self.calls += 1

        def decode(self, t):
            pass

        def memory_usage(self):
            return 0.0

        def energy_usage(self):
            return 0.0

    clock = MockClock()
    b = Benchmark(clock)
    engine = FlakyEngine(clock)
    res = b.run_stable(engine, [1], 0, runs=3)
    assert res["prefill_time"] == 20.0
