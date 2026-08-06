import numpy as np

def measure_overflow_rate(scores, threshold=65504.0, softcap=None):
    if softcap is not None and softcap > 0:
        modified_scores = softcap * np.tanh(scores / softcap)
    else:
        modified_scores = scores
    overflows = np.abs(modified_scores) > threshold
    return float(np.mean(overflows))
