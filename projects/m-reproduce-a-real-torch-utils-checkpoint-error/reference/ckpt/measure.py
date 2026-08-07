def measure_memory_time(layers, interval):
    base_mem = layers * 16.0
    if interval > 0:
        mem = (layers / interval) * 16.0 + interval * 8.0
        time_cost = float(layers) + (layers / float(interval)) * 0.25
    else:
        mem = base_mem
        time_cost = float(layers)
    return {"memory": float(mem), "time": float(time_cost)}
