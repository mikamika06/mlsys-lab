def check(workdir):
    from bakeoff.engine import BakeoffEngine
    cfg = {"seed": 42}
    eng = BakeoffEngine(cfg)
    score = eng.evaluate("baseline")
    if not isinstance(score, float):
        return {"eval_tracked": 0.0}
    return {"eval_tracked": 1.0}
