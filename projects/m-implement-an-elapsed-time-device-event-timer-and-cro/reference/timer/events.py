def measure_elapsed_time(profile):
    k = profile["kernel_ms"]
    h = profile["host_ms"]
    if profile["synced"]:
        return float(k)
    return float(h + k)
