def check(workdir):
    from bakeoff.runner import BakeoffRunner
    m = {"artifacts_match": 0.0}
    try:
        runner = BakeoffRunner({"dim": 32})
        art = runner.export_artifact("stack_a")
        if isinstance(art, dict) and "format" in art:
            m["artifacts_match"] = 1.0
    except Exception:
        pass
    return m
