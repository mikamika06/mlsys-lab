import numpy as np


def compute_percentiles(raw_data):
    ttft = np.array([r["ttft"] for r in raw_data], dtype=float)
    tpot = np.array([r["tpot"] for r in raw_data], dtype=float)
    itl = np.array([r["itl"] for r in raw_data], dtype=float)
    e2el = np.array([r["e2el"] for r in raw_data], dtype=float)

    def stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
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
