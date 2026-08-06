def detect_period(latencies):
    if not latencies:
        return 0
    mean_val = sum(latencies) / len(latencies)
    spikes = [i for i, v in enumerate(latencies) if v > mean_val * 2.5]
    if len(spikes) < 2:
        return 0
    diffs = [spikes[j] - spikes[j - 1] for j in range(1, len(spikes))]
    if not diffs:
        return 0
    from collections import Counter
    return Counter(diffs).most_common(1)[0][0]
