def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from det.trainer import measure_cost
    m = {"overhead_measured": 0.0}
    try:
        cost = measure_cost()
        if isinstance(cost, float) and cost > 1.0:
            m["overhead_measured"] = 1.0
    except Exception:
        pass
    return m
