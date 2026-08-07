import numpy as np

def measure_acceptance(preds, targets):
    matches = np.array(preds) == np.array(targets)
    return float(np.mean(matches))
