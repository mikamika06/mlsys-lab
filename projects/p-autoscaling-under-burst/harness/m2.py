def check(workdir):
    from scaler.queue import simulate_queue
    m = {"queue_modeled": 0.0}
    try:
        res = simulate_queue(10, 20, 1, 3)
        if isinstance(res, dict) and "max_queue" in res:
            m["queue_modeled"] = 1.0
    except Exception:
        pass
    return m
