import ref


def check(workdir):
    m = {"stable_median_ok": 0.0}
    try:
        from bakeoff.benchmark import Benchmark
    except Exception:
        return m

    clock = ref.VirtualClock()

    class FlakyEngine(ref.MockEngine):
        def __init__(self, c):
            super().__init__(c)
            self.runs = 0

        def prefill(self, tokens):
            self.prefill_calls += 1
            times = [1.0, 3.0, 2.0]
            self.clock.advance(times[self.runs % 3])
            self.mem += self.d_mem
            self.energy += self.d_energy
            self.runs += 1

    fe = FlakyEngine(clock)
    b = Benchmark(clock)

    try:
        res = b.run_stable(fe, [1], 5, runs=3)
        if abs(res.get("prefill_time", 0.0) - 2.0) < 1e-5:
            m["stable_median_ok"] = 1.0
    except Exception:
        pass
    return m
