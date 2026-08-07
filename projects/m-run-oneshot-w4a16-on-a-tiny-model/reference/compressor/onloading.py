import numpy as np


def simulate_onloading(weights, sequential=True):
    if sequential:
        peak = max(w.nbytes for w in weights.values())
    else:
        peak = sum(w.nbytes for w in weights.values())
    return {"peak_memory": int(peak), "sequential": bool(sequential)}
