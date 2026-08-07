def check(workdir):
    from bakeoff.engine import BakeoffEngine
    cfg = {"seed": 42}
    eng = BakeoffEngine(cfg)
    res = eng.run_benchmark(runs=3)
    for b, metrics in res.items():
        if "std_time" not in metrics or "mean_time" not in metrics:
            return {"intervals_stable": 0.0}
    return {"intervals_stable": 1.0}
