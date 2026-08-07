import ref


def check(workdir):
    m = {"run_basic_ok": 0.0}
    try:
        from bakeoff.benchmark import Benchmark
    except Exception:
        return m

    clock = ref.VirtualClock()
    engine = ref.MockEngine(clock)
    b = Benchmark(clock)

    try:
        res = b.run_basic(engine, [1, 2], 4)
        if engine.prefill_calls == 1 and engine.decode_calls == 4 and res == 4:
            m["run_basic_ok"] = 1.0
    except Exception:
        pass
    return m
