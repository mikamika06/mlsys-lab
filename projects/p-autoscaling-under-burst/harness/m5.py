def check(workdir):
    from scaler.queue import simulate_queue
    m = {"slo_held": 0.0}
    try:
        res = simulate_queue(10, 50, 2, 3)
        if isinstance(res, dict) and res.get("max_queue", 1) == 0.0:
            m["slo_held"] = 1.0
    except Exception:
        pass
    return m
