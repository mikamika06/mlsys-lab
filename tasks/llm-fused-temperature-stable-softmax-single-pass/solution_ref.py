import math
import numpy as np

def fused_softmax(logits, T):
    """Temperature-scaled, numerically-stable softmax computed in ONE online pass."""
    logits = np.asarray(logits, dtype=np.float64)
    running_max = float('-inf')
    denom = 0.0
    for x in logits:
        z = x / T
        new_max = z if z > running_max else running_max
        denom = denom * math.exp(running_max - new_max) + math.exp(z - new_max)
        running_max = new_max
    return np.array([math.exp(x / T - running_max) / denom for x in logits], dtype=np.float64)
