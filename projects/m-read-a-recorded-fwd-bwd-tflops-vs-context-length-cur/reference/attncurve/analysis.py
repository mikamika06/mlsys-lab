import numpy as np

def analyze_crossover(ctxs, baseline, custom):
    diff = custom - baseline
    idx = np.argmin(np.abs(diff))
    return int(ctxs[idx])
