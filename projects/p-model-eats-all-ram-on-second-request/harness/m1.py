import ref

def check(workdir):
    from runner.memory import ModelMemoryProfiler

    m = {"footprint_ok": 0.0}
    prof = ModelMemoryProfiler(4000, 32, 4096, 32)
    val = prof.measure_footprint()
    expected = ref.get_oracle_footprint(4000)
    if abs(val - expected) < 1e-5:
        m["footprint_ok"] = 1.0
    return m
