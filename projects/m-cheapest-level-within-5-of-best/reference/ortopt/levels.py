def select_cheapest_level(latencies, setup_costs, tolerance=0.05):
    """Select the lowest level index within tolerance of best latency."""
    best_lat = min(latencies)
    threshold = best_lat * (1.0 + tolerance)
    candidates = []
    for idx, (lat, cost) in enumerate(zip(latencies, setup_costs)):
        if lat <= threshold:
            candidates.append((cost, idx))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1]
