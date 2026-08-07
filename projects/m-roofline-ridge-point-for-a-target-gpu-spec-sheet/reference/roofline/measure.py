def measure_crossover_ratio(lat_batch1: float, lat_batch32: float) -> float:
    return lat_batch32 / max(lat_batch1, 1e-9)
