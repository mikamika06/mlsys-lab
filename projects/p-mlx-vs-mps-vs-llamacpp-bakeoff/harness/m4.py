import ref


def check(workdir):
    m = {"memory_peak_ok": 0.0, "energy_used_ok": 0.0}
    try:
        from bakeoff.benchmark import Benchmark
    except Exception:
        return m

    clock = ref.VirtualClock()
    engine = ref.MockEngine(clock)
    engine.d_mem = 10.0
    engine.d_energy = 2.0
    b = Benchmark(clock)

    try:
        res = b.run_perf(engine, [1], 5)
        if abs(res.get("memory_peak", 0.0) - 60.0) < 1e-5:
            m["memory_peak_ok"] = 1.0
        if abs(res.get("energy_used", 0.0) - 12.0) < 1e-5:
            m["energy_used_ok"] = 1.0
    except Exception:
        pass
    return m
