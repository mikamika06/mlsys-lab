def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from lora_sweep import optimizer

    m = {"pareto_found": 0.0}
    try:
        dummy = [{"rank": 8, "loss": 2.1, "cost": 100}]
        res = optimizer.find_pareto_front(dummy)
        if isinstance(res, list) and len(res) > 0:
            m["pareto_found"] = 1.0
    except Exception:
        pass
    return m
