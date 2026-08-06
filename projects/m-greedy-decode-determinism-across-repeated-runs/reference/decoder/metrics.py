def compute_match_fraction(sequences):
    if not sequences:
        return 0.0
    first = sequences[0]
    matches = sum(1 for seq in sequences if seq == first)
    return float(matches / len(sequences))


def analyze_latency_ratio(cold_latencies, reused_latencies):
    avg_cold = sum(cold_latencies) / len(cold_latencies) if cold_latencies else 1.0
    avg_reused = sum(reused_latencies) / len(reused_latencies) if reused_latencies else 1.0
    if avg_reused == 0:
        return 2.0
    return float(avg_cold / avg_reused)
