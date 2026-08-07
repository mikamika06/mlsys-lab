import numpy as np


def get_raw_dataset():
    rng = np.random.default_rng(999)
    data = []
    for _ in range(200):
        data.append({
            "ttft": float(rng.exponential(0.2)),
            "tpot": float(rng.exponential(0.05)),
            "itl": float(rng.exponential(0.02)),
            "e2el": float(rng.exponential(1.0))
        })
    return data


def compute_reference_percentiles(raw_data):
    ttft = np.array([r["ttft"] for r in raw_data], dtype=float)
    tpot = np.array([r["tpot"] for r in raw_data], dtype=float)
    itl = np.array([r["itl"] for r in raw_data], dtype=float)
    e2el = np.array([r["e2el"] for r in raw_data], dtype=float)

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99))
        }

    return {
        "ttft": stats(ttft),
        "tpot": stats(tpot),
        "itl": stats(itl),
        "e2el": stats(e2el)
    }


def compute_reference_arrivals(rate, duration, seed=42):
    rng = np.random.default_rng(seed)
    if rate <= 0:
        return []
    intervals = rng.exponential(1.0 / rate, size=int(rate * duration * 3))
    times = np.cumsum(intervals)
    times = times[times <= duration]
    return times.tolist()
