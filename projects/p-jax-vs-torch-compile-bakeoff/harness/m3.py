def check(workdir):
    from bakeoff.runner import BakeoffRunner
    m = {"dynamic_guard_ok": 0.0}
    try:
        runner = BakeoffRunner({"dim": 32})
        shapes = [(4, 32), (8, 32), (4, 32)]
        res = runner.evaluate_dynamic("stack_a", shapes)
        if isinstance(res, dict) and "recompilations" in res:
            m["dynamic_guard_ok"] = 1.0
    except Exception:
        pass
    return m
