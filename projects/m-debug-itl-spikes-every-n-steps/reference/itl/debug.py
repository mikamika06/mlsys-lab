def analyze_latencies(latencies, n):
    spikes = []
    for i, val in enumerate(latencies):
        if (i + 1) % n == 0:
            spikes.append((i, val))
    return spikes
