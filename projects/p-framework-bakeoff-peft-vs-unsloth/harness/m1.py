def check(workdir):
    from bakeoff.engine import BakeoffEngine
    cfg = {"seed": 42}
    eng = BakeoffEngine(cfg)
    d1 = eng.prepare_data()
    d2 = eng.prepare_data()
    if d1.shape != d2.shape:
        return {"parity_ok": 0.0}
    return {"parity_ok": 1.0}
