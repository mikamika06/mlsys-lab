def measure_diminishing_returns(trials_list, base_latency, min_latency):
    latencies = []
    for t in trials_list:
        lat = min_latency + (base_latency - min_latency) / (1.0 + 0.05 * t)
        latencies.append(float(lat))
    return latencies
