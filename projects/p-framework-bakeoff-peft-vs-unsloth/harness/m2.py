def check(workdir):
    from bakeoff.engine import BakeoffEngine
    cfg = {"seed": 42}
    eng = BakeoffEngine(cfg)
    m = eng.step("baseline")
    if "time" not in m or "memory" not in m:
        return {"metrics_tracked": 0.0}
    return {"metrics_tracked": 1.0}
