def check(workdir):
    from scaler.cost import measure_cold_start_cost
    m = {"cost_measured": 0.0}
    try:
        res = measure_cold_start_cost(2000, 200, 5)
        if isinstance(res, dict) and "total_time" in res and res["total_time"] == 15.0:
            m["cost_measured"] = 1.0
    except Exception:
        pass
    return m
