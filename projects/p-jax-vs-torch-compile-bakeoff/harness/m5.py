def check(workdir):
    from bakeoff.runner import BakeoffRunner
    m = {"intervals_valid": 0.0}
    try:
        runner = BakeoffRunner({"dim": 32})
        runs_a = [1.0, 1.1, 1.05, 1.08]
        runs_b = [1.02, 1.07, 1.03, 1.06]
        res = runner.compute_intervals(runs_a, runs_b)
        if isinstance(res, dict) and "overlap" in res:
            m["intervals_valid"] = 1.0
    except Exception:
        pass
    return m
