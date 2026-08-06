def instrument_loop(steps, base_mem=1024):
    out = []
    active = base_mem
    cache = 0
    peak = base_mem
    for i in range(steps):
        active += 256
        cache += 128
        peak = max(peak, active + cache)
        out.append({"step": i, "active": active, "cache": cache, "peak": peak})
    return out
