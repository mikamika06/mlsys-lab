import ref

def check(workdir):
    from ort_tune.config import RuntimeEngine
    m = {"latency_target_met": 0.0}
    cfg = {"intra_threads": 4, "enable_arena": True, "io_binding": True, "opt_level": 99}
    engine = RuntimeEngine(cfg)
    lat = engine.run(None)
    if lat < 95.0:
        m["latency_target_met"] = 1.0
    return m
