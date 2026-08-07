import ref


def check(workdir):
    m = {"setup_ok": 0.0}
    try:
        from bakeoff.benchmark import Benchmark
    except Exception:
        return m

    clock = ref.VirtualClock()
    engine = ref.MockEngine(clock)
    b = Benchmark(clock)

    try:
        b.setup_engine(engine, 1024, 4, "int4")
        if engine.setup_args == (1024, 4, "int4"):
            m["setup_ok"] = 1.0
    except Exception:
        pass
    return m
