def select_fastest_config(sweep_results):
    best_cfg = None
    min_time = float("inf")
    for item in sweep_results:
        t = float(item.get("latency", float("inf")))
        if t < min_time:
            min_time = t
            best_cfg = item.get("config")
    return best_cfg
