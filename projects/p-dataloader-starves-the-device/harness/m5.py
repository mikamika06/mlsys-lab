def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dl.loader import evaluate_utilization

    m = {"utilization_ok": 0.0}
    try:
        val = evaluate_utilization(10.0, 5.0)
        if val <= 0.6:
            m["utilization_ok"] = 1.0
    except Exception:
        pass
    return m
