import random

TRACES = [
    {"load_factor": 1.2, "preemptions": [0, 2, 5, 12, 20], "latencies": [45.0, 48.0, 60.0, 110.0, 250.0]},
    {"load_factor": 1.5, "preemptions": [3, 8, 15, 25, 40], "latencies": [52.0, 75.0, 140.0, 220.0, 410.0]},
    {"load_factor": 2.0, "preemptions": [10, 20, 35, 50, 80], "latencies": [90.0, 150.0, 280.0, 450.0, 780.0]}
]

def load_traces(raw_data):
    parsed = []
    for item in raw_data:
        parsed.append({
            "load_factor": float(item["load_factor"]),
            "preemptions": [int(x) for x in item["preemptions"]],
            "latencies": [float(x) for x in item["latencies"]]
        })
    return parsed

def compute_correlation(pree, lat):
    n = len(pree)
    if n == 0:
        return 0.0
    mean_p = sum(pree) / n
    mean_l = sum(lat) / n
    num = sum((pree[i] - mean_p) * (lat[i] - mean_l) for i in range(n))
    den_p = sum((pree[i] - mean_p) ** 2 for i in range(n))
    den_l = sum((lat[i] - mean_l) ** 2 for i in range(n))
    if den_p == 0 or den_l == 0:
        return 0.0
    return num / ((den_p * den_l) ** 0.5)

def analyze_server(trace):
    corr = compute_correlation(trace["preemptions"], trace["latencies"])
    p99 = sorted(trace["latencies"])[int(0.99 * (len(trace["latencies"]) - 1))]
    return {"correlation": round(corr, 4), "p99": p99}
