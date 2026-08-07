def check(workdir):
    m = {"optimal_window_found": 0.0}
    try:
        from batching.window import find_optimal_window
    except Exception:
        return m

    curve = {1: 12.0, 2: 15.0, 4: 22.0, 8: 35.0, 16: 60.0}
    target_slo = 40.0
    try:
        res = find_optimal_window(curve, target_slo)
    except Exception:
        return m

    if isinstance(res, dict) and "optimal_window" in res and res["optimal_window"] > 0:
        m["optimal_window_found"] = 1.0
    return m
