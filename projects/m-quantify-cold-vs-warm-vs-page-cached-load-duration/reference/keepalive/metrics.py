def estimate_load_duration(model_size_mb, state, disk_bw=500.0, cache_bw=4000.0, overhead=0.05):
    if state == "warm":
        return 0.0
    if state == "page_cached":
        return overhead + (model_size_mb / cache_bw)
    if state == "cold":
        return overhead + (model_size_mb / disk_bw)
    raise ValueError(f"unknown state: {state}")
