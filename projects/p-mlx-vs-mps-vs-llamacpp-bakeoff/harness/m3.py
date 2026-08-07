import ref


def check(workdir):
    m = {"prefill_time_ok": 0.0, "decode_time_ok": 0.0}
    try:
        from bakeoff.benchmark import Benchmark
    except Exception:
        return m

    clock = ref.VirtualClock()
    engine = ref.MockEngine(clock)
    engine.p_time = 1.0
    engine.d_time = 0.1
    b = Benchmark(clock)

    try:
        res = b.run_perf(engine, [1], 10)
        if abs(res.get("prefill_time", 0.0) - 1.0) < 1e-5:
            m["prefill_time_ok"] = 1.0
        if abs(res.get("decode_time_per_token", 0.0) - 0.1) < 1e-5:
            m["decode_time_ok"] = 1.0
    except Exception:
        pass
    return m
