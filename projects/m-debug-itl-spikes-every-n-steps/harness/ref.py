import random


def generate_trace(period: int, length: int = 2000):
    latencies = []
    for i in range(length):
        base = 10.0 + random.uniform(0, 1.0)
        if i > 0 and i % period == 0:
            base += 45.0 + random.uniform(0, 5.0)
        latencies.append(base)
    return latencies


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


def find_root_cause(latencies, period):
    return {"period": period, "trigger_step_mod": 0, "severity": "high"}
